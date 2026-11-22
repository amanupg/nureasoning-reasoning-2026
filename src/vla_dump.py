"""Dump nuVLA trajectories (global (51,3), as the benchmark scores them) for a split,
using the devkit's own VLATrajectoryProvider, so they can join the chooser's
candidate set. Output: runs_vla/<tag>_<split>.npz {names, traj}."""
import argparse, os, sys
import numpy as np
from nureasoning.nuvla.trajectory_provider import VLATrajectoryProvider, _load_metadata
from nureasoning.common.pickle_io import load_pickle

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint_dir", required=True); ap.add_argument("--split", required=True)
ap.add_argument("--tag", required=True); ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--half", default="", help="only clips of this half (artifacts/vla_halves.json): out-of-fold")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
a = ap.parse_args()
prov = VLATrajectoryProvider(checkpoint_dir=a.checkpoint_dir, num_inference_steps=a.steps)
root = f"vla_data/{a.split}"; names, trajs = [], []
clips = sorted(os.listdir(root))
if a.half:
    import json
    keep = set(json.load(open("artifacts/vla_halves.json"))[a.half]); clips = [c for c in clips if c in keep]
clips = clips[a.shard::a.nshards]
for i, c in enumerate(clips):
    cp = os.path.join(root, c); fr = _load_metadata(cp)["frames"]
    try:
        ego = load_pickle(os.path.join(cp, fr[100]["ego_state"]))
        t = prov(cp, 100, ego)
    except Exception as e:
        print("skip", c, type(e).__name__, e, flush=True); continue
    if t is None: continue
    names.append(c); trajs.append(np.asarray(t, float))
    if (i + 1) % 100 == 0: print(f"  {i+1}", flush=True)
suffix = (f"_{a.half}" if a.half else "") + (f"_s{a.shard}of{a.nshards}" if a.nshards > 1 else "")
np.savez_compressed(f"runs_vla/{a.tag}_{a.split}{suffix}.npz", names=np.array(names), traj=np.stack(trajs))
print(f"{len(names)} -> runs_vla/{a.tag}_{a.split}{suffix}.npz")
