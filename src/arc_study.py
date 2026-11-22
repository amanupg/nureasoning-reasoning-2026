"""How far does the ego travel along the route in 5 s, and what predicts it?

Runs WITHOUT the scorer: extracts (features, true arc length) per validation
clip once, caches it, then evaluates predictors offline. Arc-length error maps
almost directly to FDE, which is the whole of the human-likeness term (w=0.5).
"""
import json, math, os, sys, numpy as np
sys.path.insert(0, "probe"); import _stub_torch  # noqa
sys.path.insert(0, "recon/devkit")
from shapely.geometry import Point
from nureasoning.planning.benchmark import (
    BenchmarkConfig, build_route_line, select_key_frame_idx, _load_metadata,
    _normalize_trajectory_to_benchmark_grid)
from nureasoning.common.pickle_io import load_pickle

CFG = BenchmarkConfig(); DT = 0.1


def features(cp):
    meta = _load_metadata(cp); frames = meta.get("frames", [])
    if len(frames) < 3: return None
    ki = select_key_frame_idx(frames, CFG.key_frame_index)
    ego = load_pickle(os.path.join(cp, frames[ki]["ego_state"]))
    if not ego.trajectory_future or len(ego.trajectory_future) < 2: return None
    gt = _normalize_trajectory_to_benchmark_grid(np.array(ego.trajectory_future, float),
                                                 ego.pose, CFG)
    route = build_route_line(frames[ki].get("mission_goal"))
    v0 = math.hypot(ego.velocity.get("vx", 0.), ego.velocity.get("vy", 0.))
    a0 = float(ego.acceleration.get("ax", 0.))

    # history-derived speeds: mean speed over the last 1 s and 3 s
    h = np.array(ego.trajectory_history or [], float)
    v_h1 = v_h3 = v0; acc_h = 0.0
    if len(h) >= 4:
        d = np.linalg.norm(np.diff(h[:, :2], axis=0), axis=1) / DT
        v_h1 = float(np.mean(d[-10:])) if len(d) >= 10 else float(np.mean(d))
        v_h3 = float(np.mean(d))
        if len(d) >= 20:
            acc_h = float((np.mean(d[-5:]) - np.mean(d[-20:-15])) / (1.5))
    # true arc length travelled along the route
    if route is None or route.is_empty or route.length < 1.0:
        arc = float(np.linalg.norm(gt[-1, :2] - gt[0, :2])); has_route = 0
        lat = 0.0
    else:
        s0 = route.project(Point(gt[0, 0], gt[0, 1]))
        arc = float(route.project(Point(gt[-1, 0], gt[-1, 1])) - s0); has_route = 1
        lat = float(route.distance(Point(gt[-1, 0], gt[-1, 1])))
    cmd = (frames[ki].get("mission_goal") or {}).get("command", "?")
    return dict(clip=os.path.basename(cp), v0=v0, a0=a0, v_h1=v_h1, v_h3=v_h3,
                acc_h=acc_h, arc=arc, has_route=has_route, lat=lat, cmd=str(cmd),
                euclid=float(np.linalg.norm(gt[-1, :2] - gt[0, :2])))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "cache/validation"
    cache = "runs/arc_features.json"
    if os.path.isfile(cache):
        rows = json.load(open(cache))
    else:
        clips = sorted(os.path.join(root, d) for d in os.listdir(root)
                       if os.path.isfile(os.path.join(root, d, "metadata.json")))
        rows = [r for r in (features(c) for c in clips) if r]
        json.dump(rows, open(cache, "w"))
    R = [r for r in rows if r["has_route"]]
    print(f"clips with route: {len(R)} / {len(rows)}")
    v0 = np.array([r["v0"] for r in R]); a0 = np.array([r["a0"] for r in R])
    vh1 = np.array([r["v_h1"] for r in R]); vh3 = np.array([r["v_h3"] for r in R])
    ach = np.array([r["acc_h"] for r in R]); arc = np.array([r["arc"] for r in R])

    def rep(name, pred):
        e = arc - np.maximum(pred, 0.0)
        # FDE contribution: sqrt(arc_err^2 + lateral^2); lateral is irreducible
        lat = np.array([r["lat"] for r in R])
        fde = np.sqrt(e**2 + lat**2)
        hum = human_score(fde)
        print(f"  {name:36} |err| p50={np.median(np.abs(e)):6.2f} "
              f"p90={np.percentile(np.abs(e),90):6.2f}  FDE p50={np.median(fde):5.2f} "
              f" human={hum.mean():.4f}")
        return hum.mean()

    def human_score(fde):
        f, z = 1.0, 8.0
        t = np.clip((fde - f) / (z - f), 0, 1)
        return np.where(fde <= f, 1.0, np.where(fde >= z, 0.0, 1 - t*t*(3-2*t)))

    print("\n=== arc-length predictors (validation, clips with a route) ===")
    rep("v0 * 5  (constant velocity)", v0 * 5)
    for d in (0.0, 0.25, 0.35, 0.5, 0.75, 1.0):
        rep(f"v0*5 + 0.5*({d})*a0*25", v0*5 + 0.5*d*a0*25)
    rep("v_hist1s * 5", vh1 * 5)
    rep("v_hist3s * 5", vh3 * 5)
    rep("0.5*(v0+v_h1) * 5", 0.5*(v0+vh1)*5)
    # least squares on a small feature set
    X = np.column_stack([v0, a0, vh1, vh3, ach, np.ones_like(v0)])
    coef, *_ = np.linalg.lstsq(X, arc, rcond=None)
    rep("least squares on ego features", X @ coef)
    print(f"    coef v0={coef[0]:.3f} a0={coef[1]:.3f} vh1={coef[2]:.3f} "
          f"vh3={coef[3]:.3f} acc_h={coef[4]:.3f} b={coef[5]:.3f}")
    print(f"\n  ORACLE arc length (lateral offset only)       human="
          f"{human_score(np.array([r['lat'] for r in R])).mean():.4f}")


if __name__ == "__main__":
    main()
