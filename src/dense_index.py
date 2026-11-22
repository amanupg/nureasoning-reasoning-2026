"""2026-11-15 compact image index for the dense-frame VLM run, built where cache/train metadata is local (Mac SSD),
so training never reads ~13 GB of metadata.json over GPFS (the first smoke, 13912323, spent 1.5 h in TrainSet init).
For every clip in artifacts/dense_members_train_{A,B}.json: the frames each pass fetched, their timestamps (to match
vqa_train_dense_cl/<clip>/<timestamp_us>_vqa.json), and per frame the 8 cameras' relative paths at t-10 and t,
with the image root: img_dense_A / img_dense_B by pass; key frame 100 -> img_vlm if the clip is in
artifacts/img_vlm_train_clips.txt, else the pass that fetched 100.
Writes artifacts/dense_index_train.json: {clip: {"ts": {timestamp_us: frame}, "img": {frame: [root, [[cam, rel_t-10, rel_t] x 8]]}}}.
usage: python src/dense_index.py"""
import json, os
PASSES = os.environ.get("DENSE_PASSES", "AB"); OUT = os.environ.get("DENSE_INDEX", "artifacts/dense_index_train.json")   # 11-16: v2 = ABC
from vlm_data import CAMS
have = set(open("artifacts/img_vlm_train_clips.txt").read().split())
P = {tag: {r["clip"]: r["frames"] for r in json.load(open(f"artifacts/dense_members_train_{tag}.json"))["train"]} for tag in PASSES}
out, n = {}, 0
for c in sorted(set().union(*P.values())):
    fr = json.load(open(f"cache/train/{c}/metadata.json"))["frames"]
    roots = {t: f"img_dense_{tag}" for tag in PASSES[::-1] for t in P[tag].get(c, [])}     # earlier pass wins if both (never expected)
    if c in have: roots[100] = "img_vlm"
    img = {}
    for t, root in roots.items():
        if t - 10 < 0 or t >= len(fr): continue
        cams = [[name, fr[t - 10]["sensors"]["cameras"].get(name), fr[t]["sensors"]["cameras"].get(name)] for name, _ in CAMS]
        if all(a and b for _, a, b in cams): img[t] = [root, cams]; n += 1
    out[c] = {"ts": {fr[t]["timestamp_us"]: t for t in roots if t < len(fr)}, "img": img}
json.dump(out, open(OUT, "w"))
print(f"{len(out)} clips, {n} (clip, frame) image sets, key-frame from img_vlm {sum(c in have for c in out)}")
