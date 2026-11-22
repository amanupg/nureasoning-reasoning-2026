"""Fetch chosen members from a remote tar WITHOUT downloading the archive.

HF caps this account at ~112 MB/s total, so streaming whole tars (2.3 GB/clip)
would take ~58 h for train. We only need 16 JPEGs (~17 MB) per clip.

A tar has no index, so we walk its 512-byte headers with HTTP range requests
over ONE keep-alive TLS connection (a fresh connection per request cost 141 ms on
the laptop, mostly handshake). Each GNU @LongLink header, its name block and the
real header are contiguous, so one 1536-byte read covers all three. Then only the
wanted members' bytes are fetched. The token is used once, in-process, to resolve
the signed CDN URL; it never appears in argv.
"""
import http.client, json, os, time, urllib.parse, urllib.request
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tarrange import _valid_header, BLOCK

BASE = "https://huggingface.co/datasets/nureasoning/nuReasoning/resolve/main/"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k): return None


def cdn_url(path, token):
    op = urllib.request.build_opener(_NoRedirect)
    try:
        op.open(urllib.request.Request(BASE + path, method="HEAD",
                headers={"Authorization": f"Bearer {token}"}), timeout=60)
        raise RuntimeError("expected a redirect to the CDN")
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303, 307, 308):
            raise
        return e.headers["Location"]


class Ranger:
    def __init__(self, url):
        u = urllib.parse.urlsplit(url)
        self.host, self.path = u.netloc, u.path + ("?" + u.query if u.query else "")
        self.conn = None; self.nreq = 0; self.nbytes = 0

    def get(self, a, b):
        for attempt in range(4):
            try:
                if self.conn is None:
                    self.conn = http.client.HTTPSConnection(self.host, timeout=60)
                self.conn.request("GET", self.path, headers={"Range": f"bytes={a}-{b}"})
                r = self.conn.getresponse(); data = r.read()
                if r.status not in (200, 206):
                    raise IOError(f"HTTP {r.status}")
                self.nreq += 1; self.nbytes += len(data)
                return data
            except Exception:
                try: self.conn.close()
                except Exception: pass
                self.conn = None; time.sleep(1 + attempt)
        raise IOError(f"range {a}-{b} failed after retries")


def fetch_members(path, size, wanted, token, out_dir):
    """Walk the tar index, then download only `wanted` member names."""
    rg = Ranger(cdn_url(path, token))
    wanted = set(wanted); found = {}; pos = 0
    while pos + BLOCK <= size and len(found) < len(wanted):
        blk = rg.get(pos, min(pos + 3 * BLOCK, size) - 1)
        h = _valid_header(blk[:BLOCK])
        if h is None:
            break                                            # end-of-archive zero blocks
        name, sz, tf = h
        if tf == b"L":                                       # GNU long name: name data + real header follow
            real_name = blk[BLOCK:BLOCK + sz].split(b"\0")[0].decode("utf-8", "replace")
            hdr_pos = pos + BLOCK + ((sz + BLOCK - 1) // BLOCK) * BLOCK
            h2 = _valid_header(blk[hdr_pos - pos:hdr_pos - pos + BLOCK]) if hdr_pos - pos + BLOCK <= len(blk) else None
            if h2 is None:
                h2 = _valid_header(rg.get(hdr_pos, hdr_pos + BLOCK - 1))
            _, sz2, _ = h2
            data_pos, name, sz = hdr_pos + BLOCK, real_name, sz2
        else:
            data_pos = pos + BLOCK
        if name in wanted:
            found[name] = (data_pos, sz)
        pos = data_pos + ((sz + BLOCK - 1) // BLOCK) * BLOCK
    for name, (a, sz) in found.items():
        dst = os.path.join(out_dir, name.split("/", 1)[1])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst + ".tmp", "wb") as fh:
            fh.write(rg.get(a, a + sz - 1))
        os.replace(dst + ".tmp", dst)
    return len(found), rg.nreq, rg.nbytes


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=2); a = ap.parse_args()
    tok = open(os.path.join(os.environ["HF_HOME"], "token")).read().strip()
    rows = json.load(open("src/image_members.json"))[a.split][100:100 + a.n]
    for r in rows:
        t0 = time.time()
        got, nreq, nb = fetch_members(r["tar"], r["size"], r["members"], tok, f"img_probe/{r['clip']}")
        dt = time.time() - t0
        print(f"{r['clip'][:40]}  {got}/16 imgs  {nreq} req  {nb/1e6:.1f} MB of {r['size']/1e9:.2f} GB  "
              f"{dt:.1f}s  ({1000*dt/nreq:.1f} ms/req)", flush=True)


# ---------------------------------------------------------------------------
# Segment-parallel walk: ~95 ms per request makes a serial walk of ~1,800
# headers take ~170 s. Split the tar into K byte ranges, resync each on the
# first GNU @LongLink header after its start (every camera member here has one,
# paths are 116+ chars), and walk the K slices concurrently, each on its own
# keep-alive connection. Slice k stops once it passes slice k+1's resync point,
# so every member is covered exactly once with its full long name.
# ---------------------------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor


def _safe_start(block, h):
    """A header we can start walking from without losing a member's full name.

    These tars MIX formats: part of each archive stores long paths in the POSIX
    ustar `prefix` field (magic "ustar\\0"), part uses GNU @LongLink records
    (typeflag 'L'). Starting on a GNU-magic regular header is unsafe: it may be
    the truncated follower of a LongLink that began before our slice.
    """
    return h[2] == b"L" or block[257:263] == b"ustar\x00"


def _resync(rg, start, size, step=512 * 1024, limit=4 * 1024 * 1024):
    """Offset of the first safe header at or after `start` (512-aligned)."""
    pos = (start + BLOCK - 1) // BLOCK * BLOCK
    end = min(size, pos + limit)
    while pos < end:
        buf = rg.get(pos, min(pos + step, size) - 1)
        for o in range(0, len(buf) - BLOCK + 1, BLOCK):
            h = _valid_header(buf[o:o + BLOCK])
            if h is not None and _safe_start(buf[o:o + BLOCK], h):
                return pos + o
        pos += len(buf) - (len(buf) % BLOCK)
    return None


def _walk(rg, start, stop, size, wanted, stop_at_non_camera=True, index=None):
    found = {}; pos = start
    while pos is not None and pos < stop and pos + BLOCK <= size:
        blk = rg.get(pos, min(pos + 3 * BLOCK, size) - 1)
        h = _valid_header(blk[:BLOCK])
        if h is None:
            break
        name, sz, tf = h
        if tf == b"L":
            real = blk[BLOCK:BLOCK + sz].split(b"\0")[0].decode("utf-8", "replace")
            hp = pos + BLOCK + ((sz + BLOCK - 1) // BLOCK) * BLOCK
            h2 = _valid_header(blk[hp - pos:hp - pos + BLOCK]) if hp - pos + BLOCK <= len(blk) else None
            if h2 is None:
                h2 = _valid_header(rg.get(hp, hp + BLOCK - 1))
            name, sz, dp = real, h2[1], hp + BLOCK
        else:
            dp = pos + BLOCK
        if index is not None:
            index[name] = (dp, sz)
        if name in wanted:
            found[name] = (dp, sz)
            if not stop_at_non_camera and len(found) == len(wanted):
                break          # targeted walk: everything found
        if stop_at_non_camera and "/cameras/" not in name and not name.endswith("reasoning_questions.json"):
            break          # measured layout: every non-image member sits in the tail block
        pos = dp + ((sz + BLOCK - 1) // BLOCK) * BLOCK
    return found


def fetch_members_parallel(path, size, wanted, token, out_dir, K=16, index_path=None):
    """index_path: also save every camera member's (offset, size) seen by the walk,
    so later frame requests for this tar need no re-walk."""
    url = cdn_url(path, token)
    wanted = set(wanted)
    rgs = [Ranger(url) for _ in range(K)]
    bounds = [size * k // K for k in range(K)]
    with ThreadPoolExecutor(K) as ex:
        starts = list(ex.map(lambda k: 0 if k == 0 else _resync(rgs[k], bounds[k], size), range(K)))
    # drop failed resyncs and keep starts strictly increasing
    st = sorted({s for s in starts if s is not None} | {0})
    starts = st + [size]; K2 = len(st)
    idx = [dict() if index_path else None for _ in range(K2)]
    with ThreadPoolExecutor(K2) as ex:
        parts = list(ex.map(lambda k: _walk(rgs[k], starts[k], starts[k + 1], size, wanted, index=idx[k]), range(K2)))
    found = {}
    for p in parts: found.update(p)
    if len(found) < len(wanted):          # layout assumption broken: full serial walk
        found = _walk(rgs[0], 0, size, size, wanted, stop_at_non_camera=False)
    if index_path:
        full = {}
        for d in idx: full.update(d)
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        json.dump(full, open(index_path, "w"))
    items = list(found.items())
    def dl_group(k):                      # one connection per thread (http.client is not thread-safe)
        for name, (a, sz) in items[k::K]:
            dst = os.path.join(out_dir, name.split("/", 1)[1])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst + ".tmp", "wb") as fh:
                fh.write(rgs[k].get(a, a + sz - 1))
            os.replace(dst + ".tmp", dst)
    with ThreadPoolExecutor(K) as ex:
        list(ex.map(dl_group, range(K)))
    return len(found), sum(r.nreq for r in rgs), sum(r.nbytes for r in rgs)
