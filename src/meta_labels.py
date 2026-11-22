"""Future-derived longitudinal meta-action at the key frame (frame 100) + ego-history text features.

Target (as asked by the jepa planning thread, 2026-11-06): DriveMA thresholds on the RECORDED 5 s future
(arXiv 2605.31271; same definition as jepa-nureasoning/src/meta_oracle.py, re-implemented here, nothing imported):
  stop if v_max <= 1.333 m/s and 5 s distance <= 1.952 m; else by the least-squares speed slope k_v over 5 s:
  decelerate <= -0.448, accelerate >= 0.429, keep otherwise.  c4 in {0 stop, 1 decel, 2 keep, 3 accel};
  c9 = stop + 8 slope bins (edges -1.5, -0.8, -0.448, 0, 0.429, 0.8, 1.5).
History (inputs only, frames <= 100): speed at -5..0 s (1 s steps), longitudinal acceleration, yaw rate, command.
Test gets history only.  -> artifacts/meta_labels_{split}.json  {clip: {...}}
  .venv/bin/python src/meta_labels.py [splits]
"""
import json, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "recon", "devkit"))
KEY = 100
EDGES = [-1.5, -0.8, -0.448, 0.0, 0.429, 0.8, 1.5]


def classes(v, dist):
    t = np.arange(len(v)) * 0.1; kv = float(np.polyfit(t, v, 1)[0])
    c4 = 0 if (v.max() <= 1.333 and dist <= 1.952) else (1 if kv <= -0.448 else (3 if kv >= 0.429 else 2))
    c9 = 0 if c4 == 0 else 1 + int(np.searchsorted(EDGES, kv))
    return c4, c9, kv


def one(args):
    from nureasoning.common.pickle_io import load_pickle
    split, clip = args
    cd = os.path.join(ROOT, "cache", split, clip)
    try:
        md = json.load(open(os.path.join(cd, "metadata.json")))
    except Exception:
        return clip, None
    fr = {f["frame_index"]: f for f in md["frames"]}
    if KEY not in fr: return clip, None
    E = {}
    for k in range(KEY - 50, KEY + 51):
        if k in fr and (k <= KEY or split != "test"):
            try: E[k] = load_pickle(os.path.join(cd, fr[k]["ego_state"]))
            except Exception: pass
    if KEY not in E: return clip, None
    sp = lambda e: float(np.hypot(e.velocity["vx"], e.velocity["vy"]))
    hist = {f"v_m{s}": (sp(E[KEY - 10 * s]) if (KEY - 10 * s) in E else None) for s in range(6)}
    e0 = E[KEY]; yaw = lambda e: e.pose["yaw"]
    yr = None
    if KEY - 10 in E:
        d = (yaw(e0) - yaw(E[KEY - 10]) + np.pi) % (2 * np.pi) - np.pi; yr = float(d)
    out = dict(hist, ax=float(e0.acceleration["ax"]), yaw_rate=yr, command=fr[KEY].get("mission_goal", {}).get("command"))
    if split != "test" and all(KEY + k in E for k in range(51)):
        v = np.array([sp(E[KEY + k]) for k in range(51)])
        P = np.array([[E[KEY + k].pose["x"], E[KEY + k].pose["y"]] for k in range(51)])
        dist = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
        c4, c9, kv = classes(v, dist)
        out.update(c4=c4, c9=c9, kv=kv, dist5=dist)
    return clip, out


def main():
    splits = sys.argv[1].split(",") if len(sys.argv) > 1 else ["train", "validation", "test"]
    for split in splits:
        clips = sorted(os.listdir(os.path.join(ROOT, "cache", split)))
        with ProcessPoolExecutor(8) as ex:
            R = {c: o for c, o in ex.map(one, [(split, c) for c in clips], chunksize=16) if o is not None}
        json.dump(R, open(os.path.join(ROOT, "artifacts", f"meta_labels_{split}.json"), "w"))
        c4 = [o["c4"] for o in R.values() if "c4" in o]
        print(f"{split}: {len(R)}/{len(clips)} clips, labelled {len(c4)}; c4 (stop, decel, keep, accel) "
              f"{np.bincount(c4, minlength=4).tolist() if c4 else '-'}", flush=True)


if __name__ == "__main__":
    main()
