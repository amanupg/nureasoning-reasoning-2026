"""Ceiling of a Hydra-MDP-style trajectory vocabulary on this metric.

Every NAVSIM winner scores a fixed trajectory vocabulary with learned per-sub-metric
heads instead of regressing a trajectory. Before building that, measure the ceiling:
if an ORACLE that picks the best vocabulary entry per clip cannot beat our current
planner by much, the whole approach is not worth it.

Vocabulary here is route-conditioned and 2-D: arc length travelled x lateral offset
from the route centreline. Lateral offset is free for human-likeness (which reads
only the endpoint) but changes both gates, which is exactly where our deficit is.
"""
import os, sys, json, itertools, time
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0,HERE); sys.path.insert(0,ROOT)
sys.path.insert(0,os.path.join(ROOT,"probe")); import _stub_torch  # noqa
sys.path.insert(0,os.path.join(ROOT,"recon","devkit"))
from concurrent.futures import ProcessPoolExecutor
from shapely.geometry import Point
from nureasoning.planning.benchmark import (
    BenchmarkConfig, evaluate_clip, select_key_frame_idx, _load_metadata, build_route_line)
from nureasoning.common.pickle_io import load_pickle
import planners

CFG = BenchmarkConfig()
ARC_SCALES = [0.0, 0.25, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6]   # x kinematic estimate
LAT_OFFSETS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]            # metres from held offset


def candidates(meta, frames, ki, ego, route):
    """(label, trajectory) pairs spanning the vocabulary."""
    v0 = planners._speed(ego); a0 = planners._lon_accel(ego)
    L0 = planners.arc_length_kinematic(v0, a0)
    out = []
    for si, s in enumerate(ARC_SCALES):
        for li, lat in enumerate(LAT_OFFSETS):
            t = planners.route_follow(None, meta, frames, ki, ego, route,
                                      damp=0.75, bias=(s * L0 - L0) - 1.25,
                                      hold_lat=1.0)
            if lat != 0.0 and len(t) > 3:
                d = t[3, :2] - t[0, :2]; n = float(np.hypot(*d))
                if n > 1e-6:
                    nrm = np.array([-d[1]/n, d[0]/n])
                    w = np.clip(np.arange(51)*0.1/1.5, 0, 1)[:, None]
                    t = t.copy(); t[:, :2] = t[:, :2] + nrm*lat*w
            out.append(((si, li), t))
    return out


def one(cp):
    try:
        meta=_load_metadata(cp); fr=meta.get("frames",[])
        ki=select_key_frame_idx(fr,CFG.key_frame_index)
        ego=load_pickle(os.path.join(cp,fr[ki]["ego_state"]))
        route=build_route_line(fr[ki].get("mission_goal"))
        if route is None or route.is_empty or route.length < 1.0: return None
        rows=[]
        for key, traj in candidates(meta,fr,ki,ego,route):
            r=evaluate_clip(cp,CFG,traj)
            if r.error: continue
            rows.append({"key":list(key),"nps":r.planning_score,"col":r.s_collision,
                         "da":r.s_driveable,"prog":r.s_progress,"comf":r.s_comfort,
                         "hum":r.s_human})
        if not rows: return None
        return {"clip":os.path.basename(cp),"cands":rows}
    except Exception:
        return None


def main():
    root="cache/validation"; N=int(sys.argv[1]) if len(sys.argv)>1 else 300
    clips=sorted(os.path.join(root,d) for d in os.listdir(root)
                 if os.path.isfile(os.path.join(root,d,"metadata.json")))[:N]
    t0=time.time(); res=[]
    with ProcessPoolExecutor() as ex:
        for i,r in enumerate(ex.map(one, clips, chunksize=2)):
            if r: res.append(r)
            if (i+1)%50==0: print(f"  [{i+1}/{len(clips)}] {time.time()-t0:.0f}s", flush=True)
    json.dump(res, open("artifacts/vocab_scores.json","w"))
    K=len(ARC_SCALES)*len(LAT_OFFSETS)
    print(f"\nclips with route: {len(res)}   vocabulary size: {K}")
    best=np.array([max(c["nps"] for c in r["cands"]) for r in res])
    cur =np.array([next(c["nps"] for c in r["cands"] if c["key"]==[5,3]) for r in res])
    print(f"  current planner (scale 1.0, lat 0)      : {cur.mean():.4f}")
    print(f"  ORACLE best-in-vocabulary per clip      : {best.mean():.4f}")
    print(f"  headroom from perfect selection         : {best.mean()-cur.mean():+.4f}")
    # best single fixed candidate (no per-clip selection) = a fair lower bound
    M=np.array([[c["nps"] for c in sorted(r["cands"], key=lambda x:x['key'])] for r in res])
    j=int(np.argmax(M.mean(axis=0)))
    print(f"  best FIXED vocabulary entry             : {M.mean(axis=0)[j]:.4f} (index {j})")
    ora=np.array([[max(c[k] for c in r["cands"]) for k in ("col","da","prog","hum")] for r in res])
    print(f"  oracle per-component ceilings: col {ora[:,0].mean():.4f} da {ora[:,1].mean():.4f} "
          f"prog {ora[:,2].mean():.4f} hum {ora[:,3].mean():.4f}")
    print("\n  -> artifacts/vocab_scores.json")

if __name__=="__main__":
    main()
