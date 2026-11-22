"""Compare trajectory-vocabulary designs by their ORACLE ceiling on validation.

v63  : current. arc = scale x kinematic estimate (9) x lateral offset (7).
       Flaw: when the ego is stopped the kinematic arc is ~0, so every scale
       gives ~0 and no candidate can start moving.
abs85: absolute arc lengths 0..77 m (17) x lateral offsets (5).
union: v63 + abs85.
Ceiling = mean over clips of the best candidate's true NPS (devkit scorer).
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "probe")); import _stub_torch  # noqa
sys.path.insert(0, os.path.join(ROOT, "recon", "devkit"))
from concurrent.futures import ProcessPoolExecutor
from nureasoning.planning.benchmark import BenchmarkConfig, evaluate_clip, select_key_frame_idx, _load_metadata, build_route_line
from nureasoning.common.pickle_io import load_pickle
import planners
from vocab_labels import VOCAB as V63, build_candidate

CFG = BenchmarkConfig()
ABS_ARCS = [0, 1, 2.5, 5, 7.5, 10, 13, 16, 20, 25, 30, 36, 43, 50, 58, 67, 77]
ABS_LATS = [-1.5, -0.75, 0.0, 0.75, 1.5]


def at_arc(meta, fr, ki, ego, route, L, lat):
    K0 = planners.arc_length_kinematic(planners._speed(ego), planners._lon_accel(ego), 0.75, 0.0)
    t = planners.route_follow(None, meta, fr, ki, ego, route, damp=0.75, bias=L - K0, hold_lat=1.0)
    if lat and len(t) > 3:
        d = t[3, :2] - t[0, :2]; n = float(np.hypot(*d))
        if n > 1e-6:
            nrm = np.array([-d[1] / n, d[0] / n]); w = np.clip(np.arange(51) * 0.1 / 1.5, 0, 1)[:, None]
            t = t.copy(); t[:, :2] += nrm * lat * w
    return t


def nps(r): return r.planning_score


def one(cp):
    try:
        meta = _load_metadata(cp); fr = meta.get("frames", []); ki = select_key_frame_idx(fr, CFG.key_frame_index)
        ego = load_pickle(os.path.join(cp, fr[ki]["ego_state"])); route = build_route_line(fr[ki].get("mission_goal"))
        if route is None or route.is_empty or route.length < 1.0: return None
        L0 = planners.arc_length_kinematic(planners._speed(ego), planners._lon_accel(ego))
        a = [nps(evaluate_clip(cp, CFG, build_candidate(meta, fr, ki, ego, route, s, l, L0))) for s, l in V63]
        b = [nps(evaluate_clip(cp, CFG, at_arc(meta, fr, ki, ego, route, L, l))) for L in ABS_ARCS for l in ABS_LATS]
        v0 = planners._speed(ego)
        return {"v63": a, "abs85": b, "stopped": v0 < 0.5, "default": a[V63.index((1.0, 0.0))]}
    except Exception:
        return None


def main():
    root = "cache/validation"; N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    clips = sorted(os.path.join(root, d) for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "metadata.json")))
    rng = np.random.RandomState(1); clips = [clips[i] for i in sorted(rng.choice(len(clips), N, replace=False))]
    t0 = time.time()
    with ProcessPoolExecutor() as ex:
        R = [r for r in ex.map(one, clips, chunksize=2) if r]
    A = np.array([r["v63"] for r in R]); B = np.array([r["abs85"] for r in R]); st = np.array([r["stopped"] for r in R])
    dflt = np.array([r["default"] for r in R])
    print(f"{len(R)} random validation route clips ({st.sum()} stopped at key frame)  {time.time()-t0:.0f}s")
    for name, M in (("v63 (current)", A), ("abs85", B), ("union 148", np.hstack([A, B]))):
        o = M.max(1)
        print(f"  {name:14} oracle {o.mean():.4f}   stopped clips {o[st].mean():.4f}   moving {o[~st].mean():.4f}")
    print(f"  default cand.  {dflt.mean():.4f}   stopped {dflt[st].mean():.4f}   moving {dflt[~st].mean():.4f}")


if __name__ == "__main__":
    main()
