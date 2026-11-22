"""Local NPS harness: run the UNMODIFIED devkit scorer over cached clips.

The cache holds every non-image file, which is exactly what benchmark.py reads,
so `evaluate_clip` runs against it directly. No reimplementation of the metric.
Providers return GLOBAL-frame (51,3) [x, y, yaw], as evaluate_clip expects.
`gt` measures the ceiling: ground truth scored against itself.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "probe")); import _stub_torch  # noqa: F401
sys.path.insert(0, os.path.join(ROOT, "recon", "devkit"))

from nureasoning.planning.benchmark import (            # noqa: E402
    BenchmarkConfig, evaluate_clip, select_key_frame_idx, _load_metadata,
    build_route_line,
)
from nureasoning.common.pickle_io import load_pickle     # noqa: E402
import planners                                          # noqa: E402

STEPS, DT = 51, 0.1
T = np.arange(STEPS) * DT
CFG = BenchmarkConfig()
_KW: dict = {}


def _ctx(clip_path):
    meta = _load_metadata(clip_path)
    frames = meta.get("frames", [])
    ki = select_key_frame_idx(frames, CFG.key_frame_index)
    ego = load_pickle(os.path.join(clip_path, frames[ki]["ego_state"]))
    route = build_route_line(frames[ki].get("mission_goal"))
    return meta, frames, ki, ego, route


def p_gt(cp, *a):      return None
def p_zeros(cp, meta, frames, ki, ego, route):
    x, y, yaw = ego.pose["x"], ego.pose["y"], ego.pose["yaw"]
    return np.column_stack([np.full(STEPS, x), np.full(STEPS, y), np.full(STEPS, yaw)])
def p_cv(cp, meta, frames, ki, ego, route):
    x, y, yaw = ego.pose["x"], ego.pose["y"], ego.pose["yaw"]
    vx, vy = ego.velocity.get("vx", 0.0), ego.velocity.get("vy", 0.0)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.column_stack([x + (vx*c - vy*s)*T, y + (vx*s + vy*c)*T, np.full(STEPS, yaw)])
def p_route(cp, meta, frames, ki, ego, route):
    return planners.route_follow(cp, meta, frames, ki, ego, route, **_KW)
def p_turn(cp, meta, frames, ki, ego, route):
    return planners.constant_turn(cp, meta, frames, ki, ego, None)
def p_bayes(cp, meta, frames, ki, ego, route):
    return planners.route_bayes(cp, meta, frames, ki, ego, route)

PROVIDERS = {"gt": p_gt, "zeros": p_zeros, "constant_velocity": p_cv,
             "route_follow": p_route, "constant_turn": p_turn,
             "route_bayes": p_bayes}


def _one(args):
    cp, provider, kw = args
    global _KW
    _KW = kw
    try:
        meta, frames, ki, ego, route = _ctx(cp)
        traj = PROVIDERS[provider](cp, meta, frames, ki, ego, route)
        r = evaluate_clip(cp, CFG, traj)
    except Exception as e:
        return {"clip": os.path.basename(cp), "error": f"{type(e).__name__}: {e}"}
    if r.error:
        return {"clip": r.clip_name, "error": r.error}
    return {"clip": r.clip_name, "nps": r.planning_score, "col": r.s_collision,
            "da": r.s_driveable, "prog": r.s_progress, "comf": r.s_comfort,
            "hum": r.s_human, "fde": r.details_human.get("fde"),
            "ade": r.details_human.get("ade"),
            "prog_mode": r.details_progress.get("mode"),
            "gt_m": r.details_progress.get("gt_m")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="cache/validation")
    ap.add_argument("--provider", default="gt", choices=sorted(PROVIDERS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--damp", type=float, default=0.75)
    ap.add_argument("--bias", type=float, default=-1.25)
    ap.add_argument("--no-curv", action="store_true")
    ap.add_argument("--hold-lat", type=float, default=1.0)
    ap.add_argument("--lat-decay", type=float, default=1.5)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    clips = sorted(os.path.join(a.clips, d) for d in os.listdir(a.clips)
                   if os.path.isfile(os.path.join(a.clips, d, "metadata.json")))
    if a.limit: clips = clips[:a.limit]
    kw = {"damp": a.damp, "bias": a.bias, "curvature_limit": not a.no_curv,
          "hold_lat": a.hold_lat, "lat_decay_s": a.lat_decay} \
         if a.provider == "route_follow" else {}

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tag = ("_" + a.tag) if a.tag else ""
    out = f"runs/{stamp}_{a.provider}{tag}.jsonl"        # never overwrite
    os.makedirs("runs", exist_ok=True)
    t0 = time.time(); rows = []; errs = 0
    with open(out, "w") as fh:
        fh.write(json.dumps({"_meta": {"script": os.path.abspath(__file__),
                                       "provider": a.provider, "kwargs": kw,
                                       "clips_root": a.clips, "n": len(clips),
                                       "utc": stamp}}) + "\n")
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, r in enumerate(ex.map(_one, [(c, a.provider, kw) for c in clips],
                                         chunksize=4)):
                fh.write(json.dumps(r) + "\n")
                if "error" in r: errs += 1
                else: rows.append(r)
                if not a.quiet and (i + 1) % 250 == 0:
                    print(f"  [{i+1}/{len(clips)}] {time.time()-t0:.0f}s", flush=True)

    if not rows:
        print(f"no clips scored ({errs} errors) -> {out}"); return
    A = lambda k: float(np.mean([r[k] for r in rows]))
    fde = np.array([r["fde"] for r in rows])
    print(f"=== {a.provider}{tag}  n={len(rows)}  errors={errs}  ({time.time()-t0:.0f}s)")
    print(f"  planning_score : {A('nps'):.4f}    [ceiling 0.8894]")
    print(f"  col {A('col'):.4f}  da {A('da'):.4f}  prog {A('prog'):.4f}  "
          f"comf {A('comf'):.4f}  hum {A('hum'):.4f}")
    print(f"  FDE  p25={np.percentile(fde,25):.2f}  p50={np.percentile(fde,50):.2f}  "
          f"p75={np.percentile(fde,75):.2f}   <=1m: {100*np.mean(fde<=1):.1f}%  "
          f">=8m: {100*np.mean(fde>=8):.1f}%")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
