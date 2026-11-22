"""Build the small cached working set for a nuReasoning split, by range request.

Downloads ~0.7 MB per test clip and ~7-14 MB per train/val clip instead of the
222 MB / 456 MB-2.9 GB tars, by fetching only the non-image tail block (plus the
first 4 KB for the test questions).

Output, one directory per clip under --out, plus a provenance sidecar:
    <out>/<clip_name>/metadata.json, ego_state/*.pkl, [map.pkl, annotations/,
                      reasoning/, reasoning_questions.json]
    <out>/<clip_name>/_provenance.json

Never overwrites: a clip whose _provenance.json exists is skipped, so the run
is resumable and reruns are cheap.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tarrange import http_range, walk_partial_tar, BLOCK

REPO = "nureasoning/nuReasoning"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"

# keep everything that is not a camera image or a lidar sweep
def _wanted(name: str) -> bool:
    rel = name.split("/", 1)[-1] if "/" in name else name
    if rel.startswith("cameras/") or rel.startswith("lidar"):
        return False
    return rel.endswith((".json", ".pkl"))


def fetch_clip(path: str, size: int, token: str, out_root: str,
               tail_mb: float, max_tail_mb: float) -> dict:
    clip = os.path.basename(path)[:-4]                       # strip .tar
    dest = os.path.join(out_root, clip)
    prov_path = os.path.join(dest, "_provenance.json")
    if os.path.isfile(prov_path):
        return {"clip": clip, "status": "skipped"}

    url = f"{BASE}/{path}"
    got: dict[str, bytes] = {}
    bytes_fetched = 0

    # 1. head block: reasoning_questions.json lives in the first few KB on test
    head = http_range(url, token, 0, 8191)
    bytes_fetched += len(head)
    for name, off, sz, is_file in walk_partial_tar(head):
        if is_file and _wanted(name) and off + sz <= len(head):
            got[name.split("/", 1)[-1]] = head[off:off + sz]

    # 2. tail block, escalating until metadata.json is complete
    tail = tail_mb
    while True:
        want = int(tail * 1024 * 1024)
        start = max(0, (size - want) // BLOCK * BLOCK)       # keep 512 alignment
        buf = http_range(url, token, None, None, suffix_len=size - start)
        bytes_fetched += len(buf)
        found = {}
        for name, off, sz, is_file in walk_partial_tar(buf):
            if is_file and _wanted(name) and off + sz <= len(buf):
                found[name.split("/", 1)[-1]] = buf[off:off + sz]
        if "metadata.json" in found or start == 0:
            got.update(found)
            break
        tail *= 2
        if tail > max_tail_mb:
            return {"clip": clip, "status": "failed",
                    "error": f"metadata.json not found within {max_tail_mb} MB tail"}

    if "metadata.json" not in got:
        return {"clip": clip, "status": "failed", "error": "no metadata.json"}

    tmp = dest + ".tmp"
    for rel, data in got.items():
        p = os.path.join(tmp, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(data)
    prov = {
        "script": os.path.abspath(__file__),
        "repo": REPO,
        "source_path": path,
        "source_size": size,
        "bytes_fetched": bytes_fetched,
        "reduction": round(size / max(bytes_fetched, 1), 1),
        "files": {k: hashlib.sha256(v).hexdigest() for k, v in sorted(got.items())},
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(tmp, "_provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)
    os.replace(tmp, dest)
    return {"clip": clip, "status": "ok", "bytes": bytes_fetched,
            "n_files": len(got), "reduction": prov["reduction"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["test", "train", "validation"])
    ap.add_argument("--tree", default="recon/tree_full.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tail-mb", type=float, default=0)
    ap.add_argument("--max-tail-mb", type=float, default=96)
    a = ap.parse_args()

    out = a.out or f"cache/{a.split}"
    os.makedirs(out, exist_ok=True)
    tail = a.tail_mb or (2.0 if a.split == "test" else 24.0)

    rows = json.load(open(a.tree))
    files = [(x["path"], (x.get("lfs") or {}).get("size", x.get("size", 0)) or 0)
             for x in rows
             if x.get("type") == "file" and x["path"].startswith(f"data/{a.split}/")]
    files.sort()
    if a.limit:
        files = files[:a.limit]
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    token = open(os.path.join(hf, "token")).read().strip()   # used in-process only, never argv

    print(f"[ingest] split={a.split} clips={len(files)} workers={a.workers} "
          f"tail={tail} MB -> {out}", flush=True)
    t0 = time.time()
    done = ok = skip = fail = 0
    total_bytes = 0
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch_clip, p, s, token, out, tail, a.max_tail_mb): p
                for p, s in files}
        for fu in as_completed(futs):
            try:
                r = fu.result()
            except Exception as e:                      # never swallow: log it
                r = {"clip": os.path.basename(futs[fu]), "status": "failed",
                     "error": f"{type(e).__name__}: {e}"}
                traceback.print_exc(file=sys.stderr)
            with lock:
                done += 1
                if r["status"] == "ok":
                    ok += 1; total_bytes += r.get("bytes", 0)
                elif r["status"] == "skipped":
                    skip += 1
                else:
                    fail += 1
                    print(f"  FAIL {r['clip']}: {r.get('error')}", flush=True)
                if done % 50 == 0 or done == len(files):
                    el = time.time() - t0
                    print(f"  [{done}/{len(files)}] ok={ok} skip={skip} fail={fail} "
                          f"{total_bytes/1e6:.0f} MB  {el:.0f}s "
                          f"({done/max(el,1e-9):.1f} clips/s)", flush=True)
    print(f"[ingest] DONE ok={ok} skipped={skip} failed={fail} "
          f"fetched={total_bytes/1e6:.1f} MB in {time.time()-t0:.0f}s", flush=True)
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
