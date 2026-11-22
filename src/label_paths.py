"""Path-source candidates for the chooser, labelled with the unmodified devkit scorer.

Generalises label_a15.py to any model that outputs a path at the key frame:
  --kind a15    : per-clip npz (xyz, rot, pose) like a15_out / a15ft_out (mean of samples)
  --kind global : one npz {names, traj (N, 51, 3) global [x, y, yaw]} like vla_dump.py
The path is re-timed to 9 relative + 17 absolute arcs + its own arc length (27, identical
to label_a15) plus the raw trajectory as generated (28 candidates) so every path source enters the chooser the same way.
Output: artifacts/labels_<tag>/<split>_XXXofNNN.npz (same keys as labels_a15).
"""
import argparse, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "probe"), os.path.join(ROOT, "recon", "devkit")]
try:
    import torch  # noqa: F401
except Exception:
    import _stub_torch  # noqa: F401
from nureasoning.planning.benchmark import BenchmarkConfig, evaluate_clip
from nureasoning.submission.format import global_to_ego
from harness import _ctx
from planners import arc_length_kinematic, _speed, _lon_accel
from label_a15 import SPECS as SPECS27
# 27 candidates identical to label_a15 (note: its "own" entry re-times the path with OUR
# speed profile to the model's arc length) + "raw": the model's trajectory as generated.
SPECS = list(SPECS27) + [("raw", 0.0)]
from a15_calib import mean_global, retime

CFG = BenchmarkConfig(); SUB = np.arange(0, 51, 5)
_G = {}


def one(args):
    split, name, kind, src, with_labels = args
    cp = os.path.join("cache", split, name)
    try:
        meta, fr, ki, ego, route = _ctx(cp)
        if kind == "a15":
            z = np.load(os.path.join(src, f"{name}.npz")); g = mean_global(z)
            spread = float(np.std(z["xyz"][:, 49, :2], 0).sum())
        else:
            if src not in _G:
                d = np.load(src); _G[src] = {n: t for n, t in zip(d["names"], d["traj"])}
            g = np.asarray(_G[src][name], float); spread = 0.0
        v0 = _speed(ego); kin = arc_length_kinematic(v0, _lon_accel(ego))
        own = float(np.sum(np.hypot(*np.diff(g[:, :2], axis=0).T)))
        T, Y = [], []
        for k_, v in SPECS:
            if k_ == "raw":
                t = g
            else:
                L = kin * v if k_ == "rel" else (v if k_ == "abs" else own)
                t = retime(g, v0, max(0.0, L), ego.pose["yaw"])
            T.append(global_to_ego(t, ego.pose)[SUB])
            if with_labels:
                r = evaluate_clip(cp, CFG, t)
                if r.error: return None
                Y.append([r.s_collision, r.s_driveable, r.s_progress, r.s_comfort, r.s_human])
        return (name, np.array(T, np.float16), np.array([spread, own, kin], np.float32),
                np.array(Y, np.float32) if with_labels else None)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True); ap.add_argument("--kind", choices=["a15", "global"], required=True)
    ap.add_argument("--src", required=True, help="dir of per-clip npz (a15) or one npz (global)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.kind == "a15":
        names = sorted(f[:-4] for f in os.listdir(a.src) if f.endswith(".npz"))
    else:
        names = sorted(str(n) for n in np.load(a.src)["names"])
    names = names[a.shard::a.nshards]; wl = a.split != "test"; t0 = time.time()
    with ProcessPoolExecutor(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))) as ex:
        R = [r for r in ex.map(one, [(a.split, n, a.kind, a.src, wl) for n in names], chunksize=2) if r]
    d = f"artifacts/labels_{a.tag}"; os.makedirs(d, exist_ok=True)
    out = f"{d}/{a.split}_{a.shard:03d}of{a.nshards:03d}.npz"
    np.savez_compressed(out, names=np.array([r[0] for r in R]), traj=np.stack([r[1] for r in R]),
                        a15feat=np.stack([r[2] for r in R]),
                        vocab=np.array([[{"rel": 0.0, "abs": 1.0, "own": 2.0, "raw": 3.0}[k], v, 0.0] for k, v in SPECS], np.float32),
                        **({"labels": np.stack([r[3] for r in R])} if wl else {}))
    print(f"{a.tag} {a.split} shard {a.shard}/{a.nshards}: {len(R)}/{len(names)} -> {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
