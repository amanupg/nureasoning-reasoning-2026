"""Member list for full-recipe nuVLA training: every reasoning-annotated frame r of
each train clip and its 1 s history frame r-10, all 8 cameras (the organisers'
recipe trains on every annotated frame; we had only frames 90/100).
Tar offsets come from the saved per-clip indexes (_index.json), so no tar walk."""
import json, os, sys
split = sys.argv[1] if len(sys.argv) > 1 else "train"
base = {r["clip"]: r for r in json.load(open("src/image_members.json"))[split]}
rows, nimg, noidx = [], 0, 0
for clip in sorted(os.listdir(f"vla_data/{split}")):
    fr = json.load(open(f"cache/{split}/{clip}/metadata.json"))["frames"]
    R = [f["frame_index"] for f in fr if f.get("reasoning") and f["frame_index"] >= 10]
    F = sorted({k for r in R for k in (r, r - 10)})
    mem = sorted({f"{clip}/{p}" for k in F for p in fr[k]["sensors"]["cameras"].values()})
    rows.append({"clip": clip, "tar": base[clip]["tar"], "size": base[clip]["size"], "members": mem})
    nimg += len(mem)
    if not any(os.path.isfile(f"{d}/{split}/{clip}/_index.json") for d in ("img", "img_a15")): noidx += 1
json.dump({split: rows}, open(f"artifacts/vla_members_{split}.json", "w"))
print(f"{split}: {len(rows)} clips, {nimg} images ({nimg/len(rows):.0f}/clip), clips without saved index: {noidx}")
