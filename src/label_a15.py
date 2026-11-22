"""Alpamayo-path candidates, labelled with the unmodified devkit scorer.

Per clip: Alpamayo 1.5 mean path (4 samples) re-timed to
  9 relative arcs (ARC_SCALES x our kinematic estimate)
  + 17 absolute arcs (ABS_ARCS)
  + Alpamayo's own timing
= 27 candidates. Same 11-point ego-frame export as labels148, so the chooser can
score 148 + 27 candidates jointly. Clip-level Alpamayo features: endpoint spread
of the samples, its own arc length, our kinematic arc.
Sharded like label148: artifacts/labels_a15/<split>_XXXofNNN.npz.
"""
import argparse, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "probe"), os.path.join(ROOT, "recon", "devkit")]
try:
    import torch  # noqa: F401  (real torch on the cluster env)
except Exception:
    import _stub_torch  # noqa: F401
from nureasoning.planning.benchmark import BenchmarkConfig, evaluate_clip
from nureasoning.submission.format import global_to_ego
from harness import _ctx
from planners import arc_length_kinematic, _speed, _lon_accel
from vocab_labels import ARC_SCALES
from vocab148 import ABS_ARCS
from a15_calib import mean_global, retime

CFG = BenchmarkConfig(); SUB = np.arange(0, 51, 5)
SPECS = [("rel", s) for s in ARC_SCALES] + [("abs", L) for L in ABS_ARCS] + [("own", 0.0)]


def one(args):
    split, name, with_labels = args
    cp = os.path.join("cache", split, name)
    try:
        z = np.load(f"a15_out/nonav/{split}/{name}.npz")
        meta, fr, ki, ego, route = _ctx(cp)
        g = mean_global(z); v0 = _speed(ego)
        kin = arc_length_kinematic(v0, _lon_accel(ego))
        own = float(np.sum(np.hypot(*np.diff(g[:, :2], axis=0).T)))
        T, Y = [], []
        for kind, v in SPECS:
            L = kin * v if kind == "rel" else (v if kind == "abs" else own)
            t = retime(g, v0, max(0.0, L), ego.pose["yaw"])
            T.append(global_to_ego(t, ego.pose)[SUB])
            if with_labels:
                r = evaluate_clip(cp, CFG, t)
                if r.error: return None
                Y.append([r.s_collision, r.s_driveable, r.s_progress, r.s_comfort, r.s_human])
        spread = float(np.std(z["xyz"][:, 49, :2], 0).sum())
        return (name, np.array(T, np.float16), np.array([spread, own, kin], np.float32),
                np.array(Y, np.float32) if with_labels else None)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    names = sorted(f[:-4] for f in os.listdir(f"a15_out/nonav/{a.split}") if f.endswith(".npz"))[a.shard::a.nshards]
    wl = a.split != "test"; t0 = time.time()
    with ProcessPoolExecutor(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))) as ex:
        R = [r for r in ex.map(one, [(a.split, n, wl) for n in names], chunksize=2) if r]
    os.makedirs("artifacts/labels_a15", exist_ok=True)
    out = f"artifacts/labels_a15/{a.split}_{a.shard:03d}of{a.nshards:03d}.npz"
    np.savez_compressed(out, names=np.array([r[0] for r in R]), traj=np.stack([r[1] for r in R]),
                        a15feat=np.stack([r[2] for r in R]),
                        vocab=np.array([[1.0 if k == "abs" else (2.0 if k == "own" else 0.0), v, 0.0] for k, v in SPECS], np.float32),
                        **({"labels": np.stack([r[3] for r in R])} if wl else {}))
    print(f"{a.split} shard {a.shard}/{a.nshards}: {len(R)}/{len(names)} -> {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
