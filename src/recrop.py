"""2026-11-14: re-cut the distance-net crops with more context, from the full-resolution key frames already on the
fileset (img_vlm/<split>/<clip>/cameras/<CAM>/<file>, the frame-100 file named in cache metadata). Same geometry as
src/img_ingest2.py (square, side = max(w, h, 32 px) * K, centred on the box, 256 px bicubic), only K changes.
-> <out>/<split>/<clip>/crops/<stem>_<line>.jpg  (same ids as img_crops, so src/dist_net.py reads it unchanged)
usage: python src/recrop.py <out dir> --k 3.0 [--workers 16]
"""
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
STEMS = ("box3d_q_test", "box3d_q_validation", "box3d_ann_validation", "box3d_ann_train")


def one(job):
    src, out, items, K = job
    try: im = Image.open(src).convert("RGB")
    except Exception: return 0
    W, H = im.size; n = 0
    for cid, (x1, y1, x2, y2) in items:
        x1, x2, y1, y2 = x1 / 1000, x2 / 1000, y1 / 1000, y2 / 1000
        cx, cy = (x1 + x2) / 2 * W, (y1 + y2) / 2 * H; s = max((x2 - x1) * W, (y2 - y1) * H, 32) * K / 2
        p = os.path.join(out, f"{cid}.jpg")
        if not os.path.exists(p):
            im.crop((int(cx - s), int(cy - s), int(cx + s), int(cy + s))).resize((256, 256), Image.BICUBIC).save(p, quality=92)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--img", default="img_vlm"); a = ap.parse_args()
    jobs, meta, miss = {}, {}, 0
    for stem in STEMS:
        for i, l in enumerate(open(f"artifacts/{stem}.jsonl")):
            s = json.loads(l); key = (s["split"], s["clip"])
            if key not in meta: meta[key] = json.load(open(f"cache/{s['split']}/{s['clip']}/metadata.json"))["frames"][100]["sensors"]["cameras"]
            rel = meta[key].get(s["cam"])
            if not rel: continue
            src = os.path.join(a.img, s["split"], s["clip"], rel)
            if not os.path.isfile(src): miss += 1; continue
            od = os.path.join(a.out, s["split"], s["clip"], "crops")
            jobs.setdefault((src, od), []).append((f"{stem}_{i}", s["box"]))
    for _, od in jobs: os.makedirs(od, exist_ok=True)
    print(f"{len(jobs)} images, {sum(len(v) for v in jobs.values())} crops, {miss} samples without a key-frame image", flush=True)
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for k, n in enumerate(ex.map(one, [(s, od, it, a.k) for (s, od), it in jobs.items()], chunksize=64)):
            done += n
            if k % 20000 == 0: print(f"  {k}/{len(jobs)} images, {done} crops", flush=True)
    print(f"RECROP_DONE {done} crops -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
