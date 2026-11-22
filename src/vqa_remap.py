"""Copy devkit VQA files (vqa_train / vqa_val, generated on the laptop with absolute laptop image
paths) to <out>/<clip>/*.json with image paths remapped to the cluster image caches
(img/<split>/<clip>/cameras/..., else img_full/...). Frames whose image is absent are dropped
from camera_sequences so build_sft only sees real files. Reports coverage.
usage: python src/vqa_remap.py vqa_train train vqa_train_cl"""
import json, os, sys
src, split, out = sys.argv[1:4]
ROOTS = [f"img/{split}", f"img_full/{split}"]
P = os.path.abspath(".")
n_img = n_ok = n_docs = 0
for clip in sorted(os.listdir(src)):
    d = os.path.join(src, clip)
    if not os.path.isdir(d): continue
    os.makedirs(os.path.join(out, clip), exist_ok=True)
    for fn in os.listdir(d):
        if not fn.endswith(".json"): continue
        doc = json.load(open(os.path.join(d, fn))); n_docs += 1
        seqs = (doc.get("temporal_multiview_context") or {}).get("camera_sequences") or {}
        for cam, frs in seqs.items():
            keep = []
            for fr in frs:
                p = fr.get("image_path") or ""; n_img += 1
                rel = p.split(f"/{clip}/", 1)[1] if f"/{clip}/" in p else None
                hit = next((os.path.join(P, r, clip, rel) for r in ROOTS if rel and os.path.isfile(os.path.join(r, clip, rel))), None)
                if hit: fr["image_path"] = hit; keep.append(fr); n_ok += 1
            seqs[cam] = keep
        doc["image_paths"] = [fr["image_path"] for frs in seqs.values() for fr in frs]
        json.dump(doc, open(os.path.join(out, clip, fn), "w"))
print(f"{src}: {n_docs} docs, image refs {n_img}, resolved {n_ok} ({n_ok / max(n_img, 1):.3f})")
