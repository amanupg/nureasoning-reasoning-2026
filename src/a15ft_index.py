"""Index of usable Alpamayo fine-tuning samples: (clip, reasoning frame r) with 16-step
history, 64 future ego states and all 12 images on disk. -> artifacts/a15ft_index_<split>.json"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a15ft_data import CAMS, _load_pickle

split = sys.argv[1] if len(sys.argv) > 1 else "train"
ROOTS = [f"{d}/{split}" for d in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["img_full", "img_a15ft", "img_a15"])]


def one(clip):
    fr = json.load(open(f"cache/{split}/{clip}/metadata.json"))["frames"]
    out = []
    for f in fr:
        r = f["frame_index"]
        if not f.get("reasoning") or r < 15: continue
        try:
            ego = _load_pickle(f"cache/{split}/{clip}/{fr[r]['ego_state']}")
        except Exception:
            continue
        if r + 64 >= len(fr) or len(getattr(ego, "trajectory_history", None) or []) < 15:
            continue
        if all(any(os.path.isfile(f"{d}/{clip}/{fr[k]['sensors']['cameras'][c]}") for d in ROOTS)
               for c, _ in CAMS for k in (r - 3, r - 2, r - 1, r)):
            out.append([clip, r])
    return out


clips = sorted(os.listdir(f"cache/{split}"))
with ProcessPoolExecutor(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as ex:
    rows = [x for o in ex.map(one, clips, chunksize=16) for x in o]
json.dump(rows, open(f"artifacts/a15ft_index_{split}.json", "w"))
print(f"{split}: {len(rows)} samples from {len({c for c, _ in rows})} clips")
