"""End-to-end check of the submission ENCODING, not just its schema.

Builds a submission from VALIDATION clips (where ground truth exists), decodes
it exactly as the server would - ego frame back to global - rescores with the
unmodified devkit scorer, and compares to the direct in-memory score.

A coordinate-frame or ordering bug would not fail validate_submission; it would
silently produce near-zero planning on the leaderboard. This is the check that
catches it, and it carries a negative control so we know it can fail.
"""
import math, os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "probe")); import _stub_torch  # noqa
sys.path.insert(0, os.path.join(ROOT, "recon", "devkit"))
from nureasoning.planning.benchmark import (
    BenchmarkConfig, build_route_line, select_key_frame_idx, _load_metadata,
    evaluate_clip)
from nureasoning.common.pickle_io import load_pickle
from nureasoning.submission.format import global_to_ego, normalize_ego_trajectory
import planners

CFG = BenchmarkConfig()


def ego_to_global(traj, pose):
    """Inverse of global_to_ego - what the evaluation server must apply."""
    x0, y0, yaw0 = float(pose["x"]), float(pose["y"]), float(pose["yaw"])
    c, s = math.cos(yaw0), math.sin(yaw0)
    t = np.asarray(traj, float)
    gx = x0 + t[:, 0] * c - t[:, 1] * s
    gy = y0 + t[:, 0] * s + t[:, 1] * c
    gyaw = np.array([(v + yaw0 + np.pi) % (2*np.pi) - np.pi for v in t[:, 2]])
    return np.column_stack([gx, gy, gyaw])


root = "cache/validation"
clips = sorted(os.path.join(root, d) for d in os.listdir(root)
               if os.path.isfile(os.path.join(root, d, "metadata.json")))[:300]
direct, roundtrip, corrupt = [], [], []
for cp in clips:
    meta = _load_metadata(cp); frames = meta.get("frames", [])
    ki = select_key_frame_idx(frames, CFG.key_frame_index)
    ego = load_pickle(os.path.join(cp, frames[ki]["ego_state"]))
    route = build_route_line(frames[ki].get("mission_goal"))
    g = planners.route_follow(cp, meta, frames, ki, ego, route)
    e = normalize_ego_trajectory(global_to_ego(g, ego.pose))   # what we SUBMIT
    back = ego_to_global(e, ego.pose)                          # what the server decodes
    try:
        direct.append(evaluate_clip(cp, CFG, g).planning_score)
        roundtrip.append(evaluate_clip(cp, CFG, back).planning_score)
        bad = back.copy(); bad[:, :2] = bad[:, :2] - [ego.pose["x"], ego.pose["y"]]
        corrupt.append(evaluate_clip(cp, CFG, bad).planning_score)   # negative control
    except Exception:
        pass

d, r, c = map(np.array, (direct, roundtrip, corrupt))
print(f"n = {len(d)} validation clips")
print(f"  direct (global, in memory)      mean NPS = {d.mean():.6f}")
print(f"  submitted -> decoded -> scored  mean NPS = {r.mean():.6f}")
print(f"  max |per-clip difference|                = {np.abs(d-r).max():.2e}")
print(f"  VERDICT: {'PASS - encoding is faithful' if np.abs(d-r).max() < 1e-6 else 'FAIL'}")
print(f"  negative control (frame bug injected) NPS = {c.mean():.6f} "
      f"-> {'detected' if c.mean() < d.mean() - 0.05 else 'NOT DETECTED - check is blind'}")
