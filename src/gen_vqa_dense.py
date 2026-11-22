"""2026-11-14 dense-frame questions: the devkit generator (same wrapper and settings as src/gen_vqa.py: history 3,
stride from the clip's frame rate, camera size patch) at the extra frames of artifacts/dense_frames_train.json.
Writes <out>/<clip>/<frame timestamp>_vqa.json (one file per frame; key frame 100 stays in vqa_train_cl).
Seed per (clip, frame) so shards are reproducible. Checked 11-14: frame 100 reproduces vqa_train exactly (82 q).
usage: python src/gen_vqa_dense.py [--out vqa_train_dense_cl] [--shard i --nshards n]"""
import argparse, json, os, random, sys, zlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gen_vqa                                             # sets up paths, the torch stub and the size patch
G = gen_vqa.G; G._image_size = gen_vqa._patched_image_size
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="vqa_train_dense_cl"); ap.add_argument("--root", default="cache/train")
ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
ap.add_argument("--nshards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
a = ap.parse_args()
F = json.load(open(os.environ.get("DENSE_FRAMES", "artifacts/dense_frames_train.json"))); clips = sorted(F)[a.shard::a.nshards]
ok = fail = 0
for i, c in enumerate(clips):
    fr = None
    for t in F[c]:
        if fr is None: fr = json.load(open(os.path.join(a.root, c, "metadata.json")))["frames"]
        o = os.path.join(a.out, c, f"{fr[t]['timestamp_us']}_vqa.json")
        if os.path.isfile(o) and os.path.getsize(o) > 0:
            try: json.load(open(o)); ok += 1; continue          # resume: keep complete files, redo partial ones
            except ValueError: pass
        random.seed(zlib.crc32(f"{c}:{t}".encode()))
        try: G.process_clip(os.path.join(a.root, c), a.out, key_frame_index=t, history_frames=3, use_gemini=False, api_key=None); ok += 1
        except Exception as e: fail += 1; print(f"  FAIL {c} {t}: {type(e).__name__}: {e}", flush=True)
    if (i + 1) % 200 == 0: print(f"  [{i + 1}/{len(clips)}] ok={ok} fail={fail}", flush=True)
print(f"[gen_vqa_dense] shard {a.shard}/{a.nshards} done ok={ok} fail={fail}", flush=True)
