"""Per-clip context features, available identically for validation and test.

The scene-level questions (driving decisions, risk assessment, unsafe action)
carry no extractable geometry in their text, so they were answered by a prior
alone. Ego kinematics and route geometry at the key frame are strong signals
for exactly those: "slow down gently" vs "maintain speed" is largely readable
off current deceleration, and lateral decisions off route curvature.
"""
import math, os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "probe")); import _stub_torch  # noqa: F401
sys.path.insert(0, os.path.join(ROOT, "recon", "devkit"))
from shapely.geometry import Point                                   # noqa: E402
from nureasoning.planning.benchmark import (                         # noqa: E402
    BenchmarkConfig, build_route_line, select_key_frame_idx, _load_metadata)
from nureasoning.common.pickle_io import load_pickle                 # noqa: E402

CFG = BenchmarkConfig()
CMD = {"GO_STRAIGHT": 0, "TURN_LEFT": 1, "TURN_RIGHT": 2}


def clip_features(clip_path):
    try:
        meta = _load_metadata(clip_path)
        frames = meta.get("frames", [])
        ki = select_key_frame_idx(frames, CFG.key_frame_index)
        ego = load_pickle(os.path.join(clip_path, frames[ki]["ego_state"]))
    except Exception:
        return {}
    v = ego.velocity if isinstance(ego.velocity, dict) else {}
    a = ego.acceleration if isinstance(ego.acceleration, dict) else {}
    f = {
        "c_v": float(math.hypot(v.get("vx", 0.0), v.get("vy", 0.0))),
        "c_ax": float(a.get("ax", 0.0)),
        "c_ay": float(a.get("ay", 0.0)),
    }
    h = np.array(ego.trajectory_history or [], float)
    if len(h) >= 6:
        d = np.linalg.norm(np.diff(h[:, :2], axis=0), axis=1) / 0.1
        f["c_vh"] = float(np.mean(d))
        f["c_dv"] = float(np.mean(d[-5:]) - np.mean(d[:5]))     # speed trend
        yaws = np.unwrap(h[:, 2])
        f["c_omega"] = float((yaws[-1] - yaws[0]) / (len(yaws) * 0.1))
    mg = frames[ki].get("mission_goal") or {}
    f["c_cmd"] = CMD.get(str(mg.get("command", "")), -1)
    rl = build_route_line(mg)
    if rl is not None and not rl.is_empty and rl.length > 1.0:
        s0 = float(rl.project(Point(ego.pose["x"], ego.pose["y"])))
        f["c_route_left"] = float(rl.length - s0)
        # heading change along the next 40 m of route = turn severity ahead
        ss = np.linspace(s0, min(s0 + 40.0, rl.length), 9)
        pts = np.array([[rl.interpolate(float(s)).x, rl.interpolate(float(s)).y] for s in ss])
        d = np.diff(pts, axis=0)
        if len(d) >= 2:
            hd = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
            f["c_turn"] = float(hd[-1] - hd[0])
            f["c_curv"] = float(np.max(np.abs(np.diff(hd))))
    else:
        f["c_route_left"] = -1.0
    return f


def build_table(root, out):
    rows = {}
    names = sorted(d for d in os.listdir(root)
                   if os.path.isfile(os.path.join(root, d, "metadata.json")))
    for i, d in enumerate(names):
        rows[d] = clip_features(os.path.join(root, d))
        if (i + 1) % 250 == 0:
            print(f"  [{i+1}/{len(names)}]", flush=True)
    json.dump(rows, open(out, "w"))
    print(f"{len(rows)} clips -> {out}")


if __name__ == "__main__":
    build_table(sys.argv[1], sys.argv[2])
