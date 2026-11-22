"""Fairness check for Alpamayo 1.5: give it the same calibration our planner gets.

Our default candidate = route polyline + a kinematic speed profile whose bias was
tuned on validation. Alpamayo was scored raw. Here we separate PATH from SPEED:
  a15_raw        : mean of 4 samples, as generated
  a15path_ourL   : Alpamayo's path shape, re-timed to our kinematic arc length
  a15path_calL   : Alpamayo's path, arc length = linear blend of Alpamayo's and
                   kinematic arc, fitted on HALF the drive logs, scored on the other
  default        : our route path + kinematic arc (what we already beat with GBDT)
Scored with the unmodified devkit metric on the validation route clips.
"""
import glob, json, math, os, random, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "probe"), os.path.join(ROOT, "recon", "devkit")]
import _stub_torch  # noqa
from nureasoning.planning.benchmark import BenchmarkConfig, evaluate_clip
from harness import _ctx
from planners import arc_length_kinematic, _profile, _speed, _lon_accel
from a15_score import to_global

CFG = BenchmarkConfig()


def mean_global(z):
    xyz, rot, pose = z["xyz"], z["rot"], tuple(z["pose"])
    ang = np.arctan2(rot[:, :, 1, 0], rot[:, :, 0, 0])
    a = np.arctan2(np.sin(ang).mean(0), np.cos(ang).mean(0))
    mr = np.zeros_like(rot[0]); mr[:, 0, 0] = mr[:, 1, 1] = np.cos(a); mr[:, 1, 0] = np.sin(a); mr[:, 0, 1] = -np.sin(a); mr[:, 2, 2] = 1
    return to_global(xyz.mean(0), mr, pose)


def retime(g, v0, L, yaw0):
    """Follow path g (51, 3) with arc-length profile reaching L at 5 s."""
    xy = g[:, :2]; seg = np.hypot(*np.diff(xy, axis=0).T); s = np.concatenate([[0], np.cumsum(seg)])
    tgt = _profile(v0, L)
    if s[-1] < 1e-3:
        h = np.array([math.cos(yaw0), math.sin(yaw0)])
        P = xy[0] + tgt[:, None] * h
    else:
        d = xy[-1] - xy[-2] if seg[-1] > 1e-3 else np.array([math.cos(g[-1, 2]), math.sin(g[-1, 2])])
        d = d / (np.linalg.norm(d) or 1.0)
        ext = tgt > s[-1]
        P = np.column_stack([np.interp(tgt, s, xy[:, 0]), np.interp(tgt, s, xy[:, 1])])
        P[ext] = xy[-1] + (tgt[ext] - s[-1])[:, None] * d
    dv = np.gradient(P, axis=0); mv = np.hypot(dv[:, 0], dv[:, 1]) > 1e-3
    yaw = np.where(mv, np.arctan2(dv[:, 1], dv[:, 0]), np.nan)
    yaw[0] = yaw0
    for i in range(1, len(yaw)):
        if np.isnan(yaw[i]): yaw[i] = yaw[i - 1]
    return np.column_stack([P, np.unwrap(yaw)])


def prep(name):
    cp = os.path.join("cache/validation", name)
    meta, fr, ki, ego, route = _ctx(cp)
    g = mean_global(np.load(f"a15_out/nonav/validation/{name}.npz"))
    arc = float(np.sum(np.hypot(*np.diff(g[:, :2], axis=0).T)))
    return dict(name=name, v0=_speed(ego), kin=arc_length_kinematic(_speed(ego), _lon_accel(ego)),
                arc=arc, yaw0=ego.pose["yaw"], g=g)


def score(args):
    d, L = args
    t = retime(d["g"], d["v0"], max(0.0, L), d["yaw0"])
    r = evaluate_clip(os.path.join("cache/validation", d["name"]), CFG, t)
    return r.planning_score if not r.error else np.nan


def main():
    ps = [np.load(f) for f in sorted(glob.glob("artifacts/labels148/validation_[0-9]*.npz"))]
    names = np.concatenate([p["names"] for p in ps]); lab = np.concatenate([p["labels"] for p in ps])
    li = {n: i for i, n in enumerate(names)}
    have = [os.path.basename(f)[:-4] for f in glob.glob("a15_out/nonav/validation/*.npz")]
    todo = sorted(n for n in have if n in li)
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        D = list(ex.map(prep, todo))
        raw = np.array(list(ex.map(score, [(d, d["arc"]) for d in D])))
        ourL = np.array(list(ex.map(score, [(d, d["kin"]) for d in D])))
        nps = lambda L: L[..., 0] * L[..., 1] * (.3 * L[..., 2] + .2 * L[..., 3] + .5 * L[..., 4])
        default = np.array([nps(lab[li[d["name"]]])[38] for d in D])
        # calibrated blend, fitted on half the logs (grid on NPS), scored on the other half, 10 splits
        log_of = lambda c: "_".join(c.split("_")[:2])
        logs = sorted({log_of(d["name"]) for d in D})
        grid = [(w, b) for w in (0, .2, .4, .6, .8, 1.0) for b in (-3, -2, -1, 0, 1)]
        cache = {g: np.array(list(ex.map(score, [(d, (1 - g[0]) * d["kin"] + g[0] * d["arc"] + g[1]) for d in D])))
                 for g in grid}
    res = []
    for seed in range(10):
        random.Random(seed).shuffle(logs); h = set(logs[:len(logs) // 2])
        fit = np.array([log_of(d["name"]) in h for d in D])
        best = max(grid, key=lambda g: np.nanmean(cache[g][fit]))
        res.append((np.nanmean(cache[best][~fit]), np.nanmean(default[~fit]), best))
    print(f"validation route clips with Alpamayo output: {len(D)}")
    print(f"  default (our route path + kinematic arc) {np.nanmean(default):.4f}")
    print(f"  a15 raw (mean of 4)                      {np.nanmean(raw):.4f}")
    print(f"  a15 path + OUR kinematic arc             {np.nanmean(ourL):.4f}")
    r = np.array([x[:2] for x in res])
    print(f"  a15 path + calibrated arc (held-out)     {r[:, 0].mean():.4f}   vs default on same halves {r[:, 1].mean():.4f}"
          f"   picks {[x[2] for x in res][:4]}")
    print(f"  per-clip: a15path_ourL beats default on {np.mean(ourL > default + 1e-9):.2f}, loses on {np.mean(ourL < default - 1e-9):.2f}")


if __name__ == "__main__":
    main()
