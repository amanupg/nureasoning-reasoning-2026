"""2026-11-15 pack the dense run's training questions into one jsonl (GPFS-friendly: one sequential read at train start).
Key frame 100 from vqa_train_cl, extra frames from vqa_train_dense_cl (timestamp -> frame via artifacts/dense_index_train.json).
Only clips in the index (overlap-excluded set) and choice questions with an answer. Threaded reads.
Writes artifacts/dense_q_train.jsonl: {"clip", "frame", "qs": [question dicts]} per (clip, frame).
usage: python src/dense_qpack.py"""
import glob, json, os
from concurrent.futures import ThreadPoolExecutor
IDX = json.load(open(os.environ.get("DENSE_INDEX", "artifacts/dense_index_train.json"))); OUT = os.environ.get("DENSE_QPACK", "artifacts/dense_q_train.jsonl")  # 11-16 v2
def ok(q): return q.get("question_type", "choice") == "choice" and q.get("choices") and "answer" in q
def one(f):
    c = f.split("/")[-2]; ts = int(os.path.basename(f).split("_")[0])
    if c not in IDX: return None
    t = 100 if f.startswith("vqa_train_cl/") else IDX[c]["ts"].get(str(ts))
    if t is None or (t == 100 and not f.startswith("vqa_train_cl/")): return None
    try: qs = [q for q in json.load(open(f)).get("questions") or [] if ok(q)]
    except ValueError: return None
    return (c, t, qs) if qs else None
fs = glob.glob("vqa_train_cl/*/*_vqa.json") + glob.glob("vqa_train_dense_cl/*/*_vqa.json")
print(f"{len(fs)} files", flush=True)
n = {}
with ThreadPoolExecutor(32) as ex, open(OUT + ".tmp", "w") as out:
    for r in ex.map(one, fs, chunksize=64):
        if r: out.write(json.dumps({"clip": r[0], "frame": r[1], "qs": r[2]}) + "\n"); n[r[1] == 100] = n.get(r[1] == 100, 0) + 1
os.replace(OUT + ".tmp", OUT)
print(f"packed: key frames {n.get(True, 0)}, extra frames {n.get(False, 0)}")
