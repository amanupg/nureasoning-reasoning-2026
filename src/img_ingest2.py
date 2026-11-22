"""Key-frame image ingest via segment-parallel tar index walks (no archive download).

Measured: HF serves ~12 range requests/s per connection and scales linearly with
connections (382 req/s at 32), while total BANDWIDTH is capped at ~112 MB/s. So we
fetch ~20-60 MB per clip instead of streaming ~2.3 GB. Output is byte-identical to
a serial walk (verified with a negative control). Resumable via _done.json.
"""
import argparse, hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tarindex import fetch_members_parallel, cdn_url, Ranger


def fetch_indexed(row, idx, token, dst, K):
    """Members at known (offset, size) from a saved tar index: no header walk. Members
    the index lacks (it stops where the original walk stopped) are found by a short
    serial walk forward from the end of the indexed region."""
    url = cdn_url(row["tar"], token); rgs = [Ranger(url) for _ in range(min(K, len(row["members"])))]
    missing = [m for m in row["members"] if m not in idx]
    if missing:
        # the index has GAPS where a segment of the original walk stopped early:
        # walk each gap (and the tail) until every missing member is found
        from tarindex import _walk, BLOCK
        spans = sorted((a, sz) for a, sz in idx.values())
        gaps = []
        for (a0, s0), (a1, _) in zip(spans, spans[1:]):
            end = (a0 + s0 + BLOCK - 1) // BLOCK * BLOCK
            if a1 - end > 4 * BLOCK: gaps.append((end, a1))
        tail = (spans[-1][0] + spans[-1][1] + BLOCK - 1) // BLOCK * BLOCK
        gaps.append((tail, row["size"]))
        from tarindex import _resync
        want = set(missing); found = {}
        for g0, g1 in sorted(gaps, key=lambda g: g[0] - g[1]):          # largest gap first
            # segment-parallel walk of the gap: K slices, each resynced on a safe header
            n = max(1, min(len(rgs), (g1 - g0) // (8 << 20)))
            b = [g0 + (g1 - g0) * k // n for k in range(n)]
            with ThreadPoolExecutor(n) as ex:
                st = list(ex.map(lambda k: g0 if k == 0 else _resync(rgs[k], b[k], g1), range(n)))
            st = sorted({x for x in st if x is not None and x < g1} | {g0}) + [g1]
            with ThreadPoolExecutor(len(st) - 1) as ex:
                parts = list(ex.map(lambda k: _walk(rgs[k % len(rgs)], st[k], st[k + 1], row["size"],
                                                    want, stop_at_non_camera=False), range(len(st) - 1)))
            for p_ in parts: found.update(p_)
            if len(found) == len(want): break
        idx = dict(idx); idx.update(found)
        if not all(m in idx for m in row["members"]):
            raise IOError(f"{len(want) - len(found)} members not found in index gaps")
    # each connection (Ranger) is used by exactly ONE thread: http.client/ssl objects are
    # not thread-safe (sharing them corrupted the heap: "double free", 2026-09-26)
    def dl_group(k):
        rg = rgs[k]
        for m in row["members"][k::len(rgs)]:
            a, sz = idx[m]
            out = os.path.join(dst, m.split("/", 1)[1]); os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out + ".tmp", "wb") as fh: fh.write(rg.get(a, a + sz - 1))
            os.replace(out + ".tmp", out)
    with ThreadPoolExecutor(len(rgs)) as ex: list(ex.map(dl_group, range(len(rgs))))
    return len(row["members"]), sum(r.nreq for r in rgs), sum(r.nbytes for r in rgs)


_CROPS = None


def one(row, out_root, token, K, save_index=False, index_from=None):
    clip = row["clip"]; dst = os.path.join(out_root, clip)
    done = os.path.join(dst, "_done.json")
    if os.path.isfile(done):
        return {"clip": clip, "status": "skipped"}
    t0 = time.time(); err = ""
    idx = None
    for root in (index_from or "").split(":"):
        ip = os.path.join(root, clip, "_index.json") if root else ""
        if ip and os.path.isfile(ip):
            idx = json.load(open(ip))
            if sum(m in idx for m in row["members"]) >= 0.8 * len(row["members"]): break
            idx = None
    for attempt in range(3):
        try:
            if idx is not None:
                got, nreq, nb = fetch_indexed(row, idx, token, dst, K)
                if got == len(row["members"]): break
            got, nreq, nb = fetch_members_parallel(row["tar"], row["size"], row["members"], token, dst, K=K,
                index_path=os.path.join(dst, "_index.json") if save_index else None)
            if got == len(row["members"]): break
            err = f"found {got}/{len(row['members'])}"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"; got, nreq, nb = 0, 0, 0
        time.sleep(15 * (attempt + 1))
    ok = all(os.path.isfile(os.path.join(dst, m.split("/", 1)[1])) for m in row["members"])
    rs = os.environ.get("IMG_RESIZE")                      # e.g. "896x592": shrink on arrival
    if ok and rs:
        from PIL import Image
        W_, H_ = map(int, rs.split("x"))
        for m in row["members"]:
            p = os.path.join(dst, m.split("/", 1)[1])
            Image.open(p).convert("RGB").resize((W_, H_), Image.BILINEAR).save(p + ".tmp.jpg", quality=90)
            os.replace(p + ".tmp.jpg", p)
    cs = os.environ.get("IMG_CROPS")                       # json {member: [{id, box (0..1000)}]}: keep crops only
    if ok and cs:
        global _CROPS
        if _CROPS is None: _CROPS = json.load(open(cs))
        from PIL import Image
        os.makedirs(os.path.join(dst, "crops"), exist_ok=True)
        for m in row["members"]:
            p = os.path.join(dst, m.split("/", 1)[1]); im = Image.open(p).convert("RGB"); W, H = im.size
            for c in _CROPS.get(m, []):
                x1, y1, x2, y2 = [v / 1000 for v in c["box"]]
                cx, cy = (x1 + x2) / 2 * W, (y1 + y2) / 2 * H; s = max((x2 - x1) * W, (y2 - y1) * H, 32) * c.get("k", 1.5) / 2
                im.crop((int(cx - s), int(cy - s), int(cx + s), int(cy + s))).resize((256, 256), Image.BICUBIC).save(
                    os.path.join(dst, "crops", f"{c['id']}.jpg"), quality=92)
            os.remove(p)                                   # the full-resolution frame is not kept
    rec = {"clip": clip, "status": "ok" if ok else "failed", "sec": round(time.time() - t0, 1),
           "req": nreq, "bytes": nb, "err": "" if ok else err,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "resized": os.environ.get("IMG_RESIZE", "")}
    if ok and not cs:
        rec["sha256"] = {m: hashlib.sha256(open(os.path.join(dst, m.split("/", 1)[1]), "rb").read()).hexdigest()
                         for m in row["members"]}
        json.dump(rec, open(done, "w"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="src/image_members.json")
    ap.add_argument("--splits", default="test,validation,train")
    ap.add_argument("--out", default="img")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_PROCID", 0)))
    ap.add_argument("--nshards", type=int, default=int(os.environ.get("SLURM_NTASKS", 1)))
    ap.add_argument("--clips", type=int, default=4, help="clips in flight")
    ap.add_argument("--K", type=int, default=24, help="parallel segments per clip")
    ap.add_argument("--save-index", action="store_true", help="write each tar's camera-member index")
    ap.add_argument("--index-from", default=None, help="dir of <clip>/_index.json from an earlier walk")
    a = ap.parse_args()
    token = open(os.path.join(os.environ["HF_HOME"], "token")).read().strip()
    members = json.load(open(a.members))
    for split in a.splits.split(","):
        rows = members[split][a.shard::a.nshards]
        out = os.path.join(a.out, split)
        print(f"[img2] {split} shard {a.shard}/{a.nshards} clips={len(rows)} inflight={a.clips} K={a.K}", flush=True)
        t0 = time.time(); ok = fail = skip = 0; req = 0
        with ThreadPoolExecutor(a.clips) as ex:
            for i, r in enumerate(ex.map(lambda row: one(row, out, token, a.K, a.save_index,
                                                          ":".join(os.path.join(r_, split) for r_ in a.index_from.split(":")) if a.index_from else None), rows)):
                ok += r["status"] == "ok"; skip += r["status"] == "skipped"; fail += r["status"] == "failed"
                req += r.get("req", 0)
                if r["status"] == "failed":
                    print(f"  FAIL {r['clip']} {r['err']}", flush=True)
                if (i + 1) % 20 == 0 or i + 1 == len(rows):
                    el = time.time() - t0
                    print(f"  [{i+1}/{len(rows)}] ok={ok} skip={skip} fail={fail} {el:.0f}s "
                          f"{(ok+fail)/max(el,1)*60:.1f} clips/min {req/max(el,1):.0f} req/s", flush=True)
        print(f"[img2] {split} DONE ok={ok} skip={skip} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
