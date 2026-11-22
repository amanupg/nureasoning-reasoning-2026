"""Merge out-of-fold trajectory dumps (e.g. h0-model on h1 clips + h1-model on h0 clips)."""
import sys, numpy as np
out, parts = sys.argv[1], sys.argv[2:]
Z = [np.load(p) for p in parts]
names = np.concatenate([z["names"] for z in Z]); traj = np.concatenate([z["traj"] for z in Z])
assert len(set(names)) == len(names), "overlapping clips across folds"
np.savez_compressed(out, names=names, traj=traj); print(out, len(names))
