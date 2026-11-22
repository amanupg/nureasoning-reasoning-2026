"""Does the human's 5-second endpoint lie ON the provided route?

If yes, predicting the endpoint collapses from a 2-D problem to a 1-D one:
how far along the route do we travel in 5 s. Arc length is far easier to
predict from ego speed/accel than a free 2-D displacement, and the route is
shipped with the TEST clips, so whatever we learn here transfers.
"""
import json, os, sys, numpy as np
sys.path.insert(0, "probe"); import _stub_torch  # noqa
sys.path.insert(0, "recon/devkit")
from shapely.geometry import Point
from nureasoning.planning.benchmark import (
    BenchmarkConfig, build_route_line, select_key_frame_idx, _load_metadata,
    _normalize_trajectory_to_benchmark_grid,
)
from nureasoning.common.pickle_io import load_pickle

cfg = BenchmarkConfig()
root = sys.argv[1] if len(sys.argv) > 1 else "cache/validation"
clips = sorted(os.path.join(root, d) for d in os.listdir(root)
               if os.path.isfile(os.path.join(root, d, "metadata.json")))

lat, arc, spd, no_route, stat = [], [], [], 0, 0
for cp in clips:
    meta = _load_metadata(cp); frames = meta.get("frames", [])
    if len(frames) < 3: continue
    ki = select_key_frame_idx(frames, cfg.key_frame_index)
    ego = load_pickle(os.path.join(cp, frames[ki]["ego_state"]))
    if not ego.trajectory_future or len(ego.trajectory_future) < 2: continue
    gt = _normalize_trajectory_to_benchmark_grid(
        np.array(ego.trajectory_future, float), ego.pose, cfg)
    rl = build_route_line(frames[ki].get("mission_goal"))
    if rl is None or rl.is_empty:
        no_route += 1; continue
    end = Point(gt[-1, 0], gt[-1, 1]); start = Point(gt[0, 0], gt[0, 1])
    lat.append(rl.distance(end))
    arc.append(rl.project(end) - rl.project(start))
    v = np.hypot(ego.velocity.get("vx", 0.0), ego.velocity.get("vy", 0.0))
    spd.append(v)
    if v < 0.5: stat += 1

lat = np.array(lat); arc = np.array(arc); spd = np.array(spd)
print(f"clips with a route: {len(lat)}/{len(clips)}  (no route: {no_route})")
print(f"\n=== lateral distance from GT 5 s endpoint to the route line ===")
for q in (10, 25, 50, 75, 90, 95):
    print(f"  p{q:<3} = {np.percentile(lat, q):6.2f} m")
print(f"  fraction within 1.0 m of the route : {100*np.mean(lat <= 1.0):5.1f}%")
print(f"  fraction within 2.0 m of the route : {100*np.mean(lat <= 2.0):5.1f}%")
print(f"  fraction within 3.0 m of the route : {100*np.mean(lat <= 3.0):5.1f}%")

print(f"\n=== arc length travelled along the route in 5 s ===")
for q in (10, 50, 90):
    print(f"  p{q:<3} = {np.percentile(arc, q):7.2f} m")
print(f"  ego stationary at key frame (<0.5 m/s): {100*stat/len(spd):.1f}%")

print(f"\n=== how well does constant speed predict arc length? ===")
pred = spd * 5.0
err = arc - pred
print(f"  median signed error (actual - const-speed): {np.median(err):+6.2f} m")
print(f"  median ABS error                          : {np.median(np.abs(err)):6.2f} m")
print(f"  p90 abs error                             : {np.percentile(np.abs(err),90):6.2f} m")
print(f"  correlation(speed*5, arc)                 : {np.corrcoef(pred, arc)[0,1]:.4f}")
print(f"\n  => a PERFECT arc-length oracle placed on the route would give")
print(f"     FDE = the lateral offset above, median {np.median(lat):.2f} m")
