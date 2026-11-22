"""Full-resolution crop list (FRAME = key frame 100 by default; env CROP_FRAME=90 and CROP_K=2.0 for the
older frame of the future-motion net, written with suffix _f<frame>) for the distance regressor: for every sample in artifacts/box3d_*.jsonl
the key-frame (frame 100) image of its camera, and the box to crop (id = <stem>_<line>).
-> src/box3d_crop_members.json ({split: [{clip, tar, size, members}]}, image_members.json format)
   artifacts/box3d_crops.json ({member: [{id, box}]})
usage: python src/box3d_crops.py   (needs src/image_members.json: tar names / sizes)
"""
import json, os
FRAME, K = int(os.environ.get("CROP_FRAME", 100)), float(os.environ.get("CROP_K", 1.5))
SUF = "" if FRAME == 100 else f"_f{FRAME}"
STEMS = os.environ.get("CROP_STEMS", "box3d_q_test,box3d_q_validation,box3d_ann_validation,box3d_ann_train").split(",")
base = json.load(open("src/image_members.json"))
crops, need = {}, {}
for stem in STEMS:
    meta = {}
    for i, l in enumerate(open(f"artifacts/{stem}.jsonl")):
        s = json.loads(l); key = (s["split"], s["clip"])
        if key not in meta: meta[key] = json.load(open(f"cache/{s['split']}/{s['clip']}/metadata.json"))["frames"][FRAME]["sensors"]["cameras"]
        rel = meta[key].get(s["cam"])
        if not rel: continue
        m = f"{s['clip']}/{rel}"
        crops.setdefault(m, []).append({"id": f"{stem}_{i}", "box": s["box"], "k": K}); need.setdefault(key, set()).add(m)
out = {}
for split in ("test", "validation", "train"):
    tar = {r["clip"]: r for r in base[split]}
    out[split] = [{"clip": c, "tar": tar[c]["tar"], "size": tar[c]["size"], "members": sorted(ms)}
                  for (sp, c), ms in sorted(need.items()) if sp == split and c in tar]
    print(split, len(out[split]), "clips,", sum(len(r["members"]) for r in out[split]), "images,",
          sum(len(crops[m]) for r in out[split] for m in r["members"]), "crops")
json.dump(out, open(f"src/box3d_crop_members{SUF}.json", "w")); json.dump(crops, open(f"artifacts/box3d_crops{SUF}.json", "w"))
