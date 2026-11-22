"""ORACLE perception features from annotations (train/validation only; test has none).

Purpose: measure what perfect perception of objects would be worth to the scorer
before building any perception model. Nothing here is usable at test time; a
perception head would have to predict these from cameras.

Clip-level (ego frame at the key frame):
  lead_*  : nearest object in the ego corridor ahead (|y| < 2 m, 0 < x < 80 m):
            distance, longitudinal speed, is_static; count of objects in corridor
  near_*  : nearest object anywhere within 30 m; counts of static / moving within 30 m
Per candidate (148, using labels148 trajectories, 11 points at 0.5 s):
  clr_static : min clearance to static objects (speed < 0.5 m/s) over the horizon
  clr_moving : min clearance to moving objects, constant-velocity extrapolation
"""
import argparse, glob, math, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "probe"), os.path.join(ROOT, "recon", "devkit")]
import _stub_torch  # noqa
from nureasoning.common.pickle_io import load_pickle
from harness import _ctx

SKIP = ("other.other", "construction_zone_area")
T11 = np.arange(11) * 0.5
CLIP_F = ["lead_d", "lead_v", "lead_static", "n_corr", "near_d", "n_static30", "n_moving30"]


def objects_ego(cp, fr, ki, ego):
    a = load_pickle(os.path.join(cp, fr[ki]["annotations"]))
    x0, y0, yaw = ego.pose["x"], ego.pose["y"], ego.pose["yaw"]
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for o in a.objects:
        if o.category in SKIP: continue
        dx, dy = o.pose["x"] - x0, o.pose["y"] - y0
        ex, ey = c * dx + s * dy, -s * dx + c * dy
        vx, vy = o.velocity.get("vx", 0.0), o.velocity.get("vy", 0.0)
        evx, evy = c * vx + s * vy, -s * vx + c * vy
        r = 0.5 * math.hypot(o.dimensions.get("l", 1.0), o.dimensions.get("w", 1.0))
        out.append((ex, ey, evx, evy, r, math.hypot(vx, vy) < 0.5))
    return np.array(out, float).reshape(-1, 6)


def one(args):
    split, name, traj = args
    cp = os.path.join("cache", split, name)
    try:
        meta, fr, ki, ego, route = _ctx(cp)
        O = objects_ego(cp, fr, ki, ego)
    except Exception:
        return name, None, None
    big = 1e3
    f = dict(lead_d=big, lead_v=0.0, lead_static=0.0, n_corr=0.0, near_d=big, n_static30=0.0, n_moving30=0.0)
    if len(O):
        d = np.hypot(O[:, 0], O[:, 1])
        corr = (O[:, 0] > 0) & (O[:, 0] < 80) & (np.abs(O[:, 1]) < 2.0)
        if corr.any():
            i = np.where(corr)[0][np.argmin(O[corr, 0])]
            f.update(lead_d=O[i, 0], lead_v=O[i, 2], lead_static=O[i, 5], n_corr=corr.sum())
        f.update(near_d=d.min(), n_static30=((d < 30) & (O[:, 5] > 0)).sum(),
                 n_moving30=((d < 30) & (O[:, 5] == 0)).sum())
    clr = np.full((traj.shape[0], 2), 50.0, np.float32)
    if len(O):
        P = traj[:, :, :2].astype(np.float32)                         # (148, 11, 2)
        for k, mask in ((0, O[:, 5] > 0), (1, O[:, 5] == 0)):
            if not mask.any(): continue
            Q = O[mask]
            ox = Q[:, 0][None, :] + Q[:, 2][None, :] * T11[:, None] * (k == 1)   # (11, n)
            oy = Q[:, 1][None, :] + Q[:, 3][None, :] * T11[:, None] * (k == 1)
            dd = np.hypot(P[:, :, 0, None] - ox[None], P[:, :, 1, None] - oy[None]) - Q[:, 4][None, None]
            clr[:, k] = np.minimum(dd.min(axis=(1, 2)), 50.0)
    return name, np.array([f[k] for k in CLIP_F], np.float32), clr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--kf", default="")        # "" = key-frame shards only
    a = ap.parse_args()
    pat = f"artifacts/labels148/{a.split}_[0-9]*.npz"
    ps = [np.load(f) for f in sorted(glob.glob(pat))]
    names = np.concatenate([p["names"] for p in ps]); traj = np.concatenate([p["traj"] for p in ps])
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        R = list(ex.map(one, [(a.split, n, t) for n, t in zip(names, traj)], chunksize=8))
    ok = [r for r in R if r[1] is not None]
    out = f"artifacts/percfeat_{a.split}.npz"
    np.savez_compressed(out, names=np.array([r[0] for r in ok]), clip=np.stack([r[1] for r in ok]),
                        clr=np.stack([r[2] for r in ok]), clip_cols=np.array(CLIP_F))
    print(f"{a.split}: {len(ok)}/{len(R)} clips -> {out}")


if __name__ == "__main__":
    main()
