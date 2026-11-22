"""Frames for Alpamayo 1.5 fine-tuning: at every reasoning-annotated frame r (with
>= 1.6 s of ego history, r >= 16), the 4-frame 10 Hz window r-3..r of the three
forward cameras. Frames r and r-10 already arrive with the full nuVLA fetch, so this
list holds only r-1, r-2, r-3. Train clips; fetched via the saved tar indexes."""
import json, os
CAMS = ["front_left", "front", "front_right"]
base = {r["clip"]: r for r in json.load(open("src/image_members.json"))["train"]}
rows, n = [], 0
for clip in sorted(os.listdir("vla_data/train")):
    fr = json.load(open(f"cache/train/{clip}/metadata.json"))["frames"]
    R = [f["frame_index"] for f in fr if f.get("reasoning") and f["frame_index"] >= 16]
    mem = sorted({f"{clip}/{fr[k]['sensors']['cameras'][c]}" for r in R for k in (r - 1, r - 2, r - 3) for c in CAMS})
    rows.append({"clip": clip, "tar": base[clip]["tar"], "size": base[clip]["size"], "members": mem}); n += len(mem)
json.dump({"train": rows}, open("artifacts/a15ft_members_train.json", "w"))
print(len(rows), "clips", n, "images", round(n / len(rows)), "per clip")
