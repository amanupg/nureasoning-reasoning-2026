"""Extract (features, true 5 s arc length) pairs for the arc-length model.

Human-likeness is w=0.5 of NPS and reads ONLY the endpoint; the human endpoint
sits a median 0.39 m from the provided route. So the entire planning problem
reduces to one scalar per clip: how far along the route the ego travels in 5 s.
This builds the supervised dataset for that scalar.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "probe")); import _stub_torch  # noqa: F401
sys.path.insert(0, os.path.join(ROOT, "recon", "devkit"))
from shapely.geometry import Point                                    # noqa: E402
from nureasoning.planning.benchmark import (                          # noqa: E402
    BenchmarkConfig, build_route_line, select_key_frame_idx, _load_metadata,
    _normalize_trajectory_to_benchmark_grid)
from nureasoning.common.pickle_io import load_pickle                  # noqa: E402

CFG = BenchmarkConfig(); DT = 0.1
CMD = {"GO_STRAIGHT": 0, "TURN_LEFT": 1, "TURN_RIGHT": 2}
LOOKAHEAD = (10.0, 20.0, 30.0, 45.0, 60.0, 80.0)


def route_geometry(route, s0):
    """Heading change and max curvature over several lookahead distances."""
    out = {}
    if route is None or route.is_empty or route.length - s0 < 2.0:
        for L in LOOKAHEAD:
            out[f"turn{int(L)}"] = 0.0
            out[f"curv{int(L)}"] = 0.0
        out["route_left"] = -1.0
        return out
    out["route_left"] = float(route.length - s0)
    for L in LOOKAHEAD:
        end = min(s0 + L, route.length)
        if end - s0 < 2.0:
            out[f"turn{int(L)}"] = 0.0; out[f"curv{int(L)}"] = 0.0; continue
        ss = np.linspace(s0, end, 11)
        pts = np.array([[route.interpolate(float(s)).x,
                         route.interpolate(float(s)).y] for s in ss])
        d = np.diff(pts, axis=0)
        hd = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        out[f"turn{int(L)}"] = float(hd[-1] - hd[0])
        seg = np.linalg.norm(d, axis=1).mean() + 1e-6
        out[f"curv{int(L)}"] = float(np.max(np.abs(np.diff(hd))) / seg)
    return out


def features_from(meta, frames, ki, ego):
    """Inference-time features: everything except the ground-truth arc length."""
    return _feats(meta, frames, ki, ego)


def one(cp):
    try:
        meta = _load_metadata(cp); frames = meta.get("frames", [])
        if len(frames) < 3: return None
        ki = select_key_frame_idx(frames, CFG.key_frame_index)
        ego = load_pickle(os.path.join(cp, frames[ki]["ego_state"]))
    except Exception:
        return None
    f = _feats(meta, frames, ki, ego)
    gtf = getattr(ego, "trajectory_future", None)
    if not gtf or len(gtf) < 2: return None
    gt = _normalize_trajectory_to_benchmark_grid(np.array(gtf, float), ego.pose, CFG)
    mg = frames[ki].get("mission_goal") or {}
    route = build_route_line(mg)
    has_route = bool(f["has_route"])
    s0 = float(route.project(Point(ego.pose["x"], ego.pose["y"]))) if has_route else 0.0
    if has_route:
        f["arc"] = float(route.project(Point(gt[-1, 0], gt[-1, 1])) - s0)
        f["lat"] = float(route.distance(Point(gt[-1, 0], gt[-1, 1])))
    else:
        f["arc"] = float(np.linalg.norm(gt[-1, :2] - gt[0, :2])); f["lat"] = 0.0
    f["euclid"] = float(np.linalg.norm(gt[-1, :2] - gt[0, :2]))
    f["clip"] = os.path.basename(cp)
    return f


def _feats(meta, frames, ki, ego):
    v = ego.velocity if isinstance(ego.velocity, dict) else {}
    a = ego.acceleration if isinstance(ego.acceleration, dict) else {}
    f = {"v0": float(math.hypot(v.get("vx", 0.), v.get("vy", 0.))),
         "ax": float(a.get("ax", 0.)), "ay": float(a.get("ay", 0.))}

    h = np.array(ego.trajectory_history or [], float)
    if len(h) >= 6:
        d = np.linalg.norm(np.diff(h[:, :2], axis=0), axis=1) / DT
        f["vh_all"] = float(np.mean(d)); f["vh_last5"] = float(np.mean(d[-5:]))
        f["vh_last10"] = float(np.mean(d[-10:])) if len(d) >= 10 else f["vh_last5"]
        f["vh_min"] = float(np.min(d)); f["vh_max"] = float(np.max(d))
        f["dv_recent"] = float(np.mean(d[-5:]) - np.mean(d[-15:-10])) if len(d) >= 15 else 0.0
        f["dv_all"] = float(d[-1] - d[0])
        f["frac_slow"] = float(np.mean(d < 0.5))
        yaws = np.unwrap(h[:, 2])
        f["omega"] = float((yaws[-1] - yaws[-min(11, len(yaws))]) / (min(10, len(yaws)-1) * DT))
    else:
        for k in ("vh_all","vh_last5","vh_last10","vh_min","vh_max"): f[k] = f["v0"]
        f["dv_recent"] = f["dv_all"] = f["omega"] = 0.0; f["frac_slow"] = 0.0

    mg = frames[ki].get("mission_goal") or {}
    f["cmd"] = CMD.get(str(mg.get("command", "")), -1)
    route = build_route_line(mg)
    has_route = route is not None and not route.is_empty and route.length >= 1.0
    s0 = float(route.project(Point(ego.pose["x"], ego.pose["y"]))) if has_route else 0.0
    f.update(route_geometry(route if has_route else None, s0))
    f["has_route"] = int(has_route)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    clips = sorted(os.path.join(a.root, d) for d in os.listdir(a.root)
                   if os.path.isfile(os.path.join(a.root, d, "metadata.json")))
    if a.limit: clips = clips[:a.limit]
    rows = []; t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor() as ex:
        for i, r in enumerate(ex.map(one, clips, chunksize=8)):
            if r: rows.append(r)
            if (i+1) % 1000 == 0:
                print(f"  [{i+1}/{len(clips)}] kept={len(rows)} {time.time()-t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rows, open(a.out, "w"))
    print(f"{len(rows)}/{len(clips)} clips -> {a.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
