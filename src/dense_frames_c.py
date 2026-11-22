"""2026-11-16 pass C of the dense-frame data: every remaining annotated 1 Hz frame per train clip (40, 60, 80, 90, 110,
120, 140; nearest annotated within 4 frames, as src/dense_frames.py), so with passes A/B and the key frame each clip has
all 13 annotated frames 30..150 (the leader's coverage: every 1 Hz reasoning frame). Same clips as passes A/B (the
overlap-excluded set in artifacts/dense_members_train.json); frames already used by A/B/key are skipped.
Writes artifacts/dense_members_train_C.json (img_ingest2 format) and artifacts/dense_frames_train_C.json.
Does NOT touch the A/B files (the running training jobs read those at restart).
usage: python src/dense_frames_c.py"""
import json
FRC = (40, 60, 80, 90, 110, 120, 140)
rows0 = json.load(open("artifacts/dense_members_train.json"))["train"]
F0 = json.load(open("artifacts/dense_frames_train.json"))
rows, frames, nimg, nfr = [], {}, 0, 0
for r in rows0:
    c = r["clip"]; fr = json.load(open(f"cache/train/{c}/metadata.json"))["frames"]
    ann = {f["frame_index"] for f in fr if f.get("reasoning")}
    used = set(F0.get(c, [])) | {100} | set(r["frames"])
    G = []
    for t in FRC:
        n = min(ann, key=lambda a: (abs(a - t), a)) if ann else None
        if n is None or abs(n - t) > 4 or n < 10 or n >= len(fr) or n in used or n in G: continue
        if all(fr[k]["sensors"]["cameras"].get(cam) for k in (n, n - 10) for cam in fr[n]["sensors"]["cameras"]): G.append(n)
    if not G: continue
    mem = sorted({f"{c}/{p}" for t in G for k in (t, t - 10) for p in fr[k]["sensors"]["cameras"].values()})
    rows.append({"clip": c, "tar": r["tar"], "size": r["size"], "members": mem, "frames": G})
    frames[c] = G; nimg += len(mem); nfr += len(G)
json.dump({"train": rows}, open("artifacts/dense_members_train_C.json", "w"))
json.dump(frames, open("artifacts/dense_frames_train_C.json", "w"))
print(f"pass C: {len(rows)} clips, {nfr} new frames, {nimg} images ({nimg / max(1, len(rows)):.0f}/clip)")
