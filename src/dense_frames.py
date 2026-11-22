"""2026-11-14 dense-frame VLM run (spec: ../reports/dense_vlm_run_spec.md): which extra train frames, which images.
Frames per clip: 30, 50, 70, 130, 150 (annotated every 10 frames from 30 to 150; Driving-decision frames are ~50/100/150,
so 50 and 150 add decision / unsafe / action-risk questions); all >= 2 s apart and >= 1 s from the key frame 100.
Clips annotated at an offset grid (31, 41, ... or 29, 39, ...) use the nearest annotated frame (within 4).
Each frame t needs all 8 cameras at t and t-10 (1 s earlier), as the VLM input. Clips without key-frame images in
img_vlm also get frames 100 / 90. Exclusions (rule copied from jepa-nureasoning/src/plan_rows.py, read-only): train clips
whose name is a validation/test clip, or whose time span overlaps a validation/test clip of the same log within 10 s.
Order: clips with img_vlm key frames first (artifacts/img_vlm_train_clips.txt), then the rest.
Writes artifacts/dense_members_train.json ({"train": [{clip, tar, size, members, frames}]}, img_ingest2 format) and
artifacts/dense_frames_train.json ({clip: [frame indices for question generation]}).
usage: python src/dense_frames.py"""
import collections, json, os
FR, KEY, MARGIN = (30, 50, 70, 130, 150), 100, 10_000_000
C = "cache"
def log_of(c): return "_".join(c.split("_")[:2])
def span(s, c):
    fr = json.load(open(f"{C}/{s}/{c}/metadata.json"))["frames"]; return fr[0]["timestamp_us"], fr[-1]["timestamp_us"]
S = {s: sorted(d for d in os.listdir(f"{C}/{s}") if os.path.isfile(f"{C}/{s}/{d}/metadata.json")) for s in ("train", "validation", "test")}
held = collections.defaultdict(list)
for s in ("validation", "test"):
    for c in S[s]: held[log_of(c)].append((s, span(s, c)))
names = set(S["validation"]) | set(S["test"]); excl = collections.Counter(); keep = []
for c in S["train"]:
    if c in names: excl["same_name"] += 1; continue
    lg = log_of(c)
    if lg in held:
        a0, b0 = span("train", c)
        hit = [s for s, (x, y) in held[lg] if a0 - MARGIN <= y and x <= b0 + MARGIN]
        if hit: excl["time_overlap_" + hit[0]] += 1; continue
    keep.append(c)
have = set(open("artifacts/img_vlm_train_clips.txt").read().split())
base = {r["clip"]: r for r in json.load(open("artifacts/image_members.json"))["train"]}
rows, frames, nimg, short = [], {}, 0, collections.Counter()
for c in sorted(keep, key=lambda c: (c not in have, c)):
    if c not in base: short["no_tar"] += 1; continue
    fr = json.load(open(f"{C}/train/{c}/metadata.json"))["frames"]
    ann = {f["frame_index"] for f in fr if f.get("reasoning")}
    F = []                                   # nearest annotated frame to each target (clips are annotated at 30+10k or offset by 1-4)
    for t in FR:
        n = min(ann, key=lambda a: (abs(a - t), a)) if ann else None
        if n is not None and abs(n - t) <= 4 and n >= 10 and n < len(fr) and n not in F: F.append(n)
    if len(F) < len(FR): short["missing_frames"] += 1
    G = F + ([KEY] if c not in have and KEY < len(fr) else [])
    mem = sorted({f"{c}/{p}" for t in G for k in (t, t - 10) for p in fr[k]["sensors"]["cameras"].values()})
    rows.append({"clip": c, "tar": base[c]["tar"], "size": base[c]["size"], "members": mem, "frames": G})
    frames[c] = F; nimg += len(mem)
json.dump({"train": rows}, open("artifacts/dense_members_train.json", "w"))
# two fetch passes so a partial fetch still covers every clip: A = the decision frames (~50, ~150) + missing key frames,
# B = the other three (~30, ~70, ~130). Same rows, members split by frame.
for tag, keepf in (("A", lambda t, F: t == KEY or (F.index(t) if t in F else -1) in (1, 4) or abs(t - 50) <= 4 or abs(t - 150) <= 4),
                   ("B", lambda t, F: t != KEY and not (abs(t - 50) <= 4 or abs(t - 150) <= 4))):
    out = []
    for r in rows:
        fr = json.load(open(f"{C}/train/{r['clip']}/metadata.json"))["frames"]
        G = [t for t in r["frames"] if keepf(t, r["frames"])]
        mem = sorted({f"{r['clip']}/{p}" for t in G for k in (t, t - 10) for p in fr[k]["sensors"]["cameras"].values()})
        if mem: out.append(dict(r, members=mem, frames=G))
    json.dump({"train": out}, open(f"artifacts/dense_members_train_{tag}.json", "w"))
    print(f"pass {tag}: {len(out)} clips, {sum(len(r['members']) for r in out)} images")
json.dump(frames, open("artifacts/dense_frames_train.json", "w"))
print(f"train {len(S['train'])}, excluded {dict(excl)}, kept {len(keep)}, with img_vlm {sum(c in have for c in keep)}; "
      f"{len(rows)} clips, {nimg} images ({nimg / max(1, len(rows)):.0f}/clip), {sum(map(len, frames.values()))} question frames; {dict(short)}")
