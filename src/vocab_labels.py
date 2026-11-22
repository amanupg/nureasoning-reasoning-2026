"""Offline label generation for a Hydra-MDP-style trajectory scorer.

For every clip, score EVERY vocabulary candidate with the unmodified devkit
scorer and record its per-sub-metric outcome. This is exactly the offline
rule-based-teacher simulation step that every NAVSIM winner performs, and we
can do it because we own the scorer.

Literature constraint honoured here: we store the SUB-METRICS separately, never
the aggregate. Distilling the aggregate PDM score measured WORSE than distilling
its components (80.2 vs 83.0 in Hydra-MDP Table 1).
"""
import argparse, json, os, sys, time
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0,HERE); sys.path.insert(0,ROOT)
sys.path.insert(0,os.path.join(ROOT,"probe")); import _stub_torch  # noqa
sys.path.insert(0,os.path.join(ROOT,"recon","devkit"))
from concurrent.futures import ProcessPoolExecutor
from nureasoning.planning.benchmark import (
    BenchmarkConfig, evaluate_clip, select_key_frame_idx, _load_metadata, build_route_line)
from nureasoning.common.pickle_io import load_pickle
import planners
from arc_data import features_from

CFG = BenchmarkConfig()
ARC_SCALES  = [0.0, 0.25, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6]
LAT_OFFSETS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
VOCAB = [(s, l) for s in ARC_SCALES for l in LAT_OFFSETS]


def build_candidate(meta, frames, ki, ego, route, scale, lat, L0):
    t = planners.route_follow(None, meta, frames, ki, ego, route,
                              damp=0.75, bias=(scale * L0 - L0) - 1.25, hold_lat=1.0)
    if lat != 0.0 and len(t) > 3:
        d = t[3, :2] - t[0, :2]; n = float(np.hypot(*d))
        if n > 1e-6:
            nrm = np.array([-d[1]/n, d[0]/n])
            w = np.clip(np.arange(51)*0.1/1.5, 0, 1)[:, None]
            t = t.copy(); t[:, :2] = t[:, :2] + nrm*lat*w
    return t


def one(cp):
    try:
        meta=_load_metadata(cp); fr=meta.get("frames",[])
        if len(fr) < 3: return None
        ki=select_key_frame_idx(fr,CFG.key_frame_index)
        ego=load_pickle(os.path.join(cp,fr[ki]["ego_state"]))
        route=build_route_line(fr[ki].get("mission_goal"))
        if route is None or route.is_empty or route.length < 1.0: return None
        feats = features_from(meta, fr, ki, ego)
        L0 = planners.arc_length_kinematic(planners._speed(ego), planners._lon_accel(ego))
        out=[]
        for (scale, lat) in VOCAB:
            r = evaluate_clip(cp, CFG, build_candidate(meta,fr,ki,ego,route,scale,lat,L0))
            if r.error: return None
            out.append([r.s_collision, r.s_driveable, r.s_progress, r.s_comfort, r.s_human])
        return {"clip": os.path.basename(cp), "f": feats, "y": out, "L0": L0}
    except Exception:
        return None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a=ap.parse_args()
    clips=sorted(os.path.join(a.root,d) for d in os.listdir(a.root)
                 if os.path.isfile(os.path.join(a.root,d,"metadata.json")))
    if a.limit: clips=clips[:a.limit]
    t0=time.time(); rows=[]
    with ProcessPoolExecutor() as ex:
        for i,r in enumerate(ex.map(one, clips, chunksize=2)):
            if r: rows.append(r)
            if (i+1)%250==0:
                print(f"  [{i+1}/{len(clips)}] kept={len(rows)} {time.time()-t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"vocab": VOCAB, "rows": rows}, open(a.out,"w"))
    print(f"{len(rows)}/{len(clips)} clips x {len(VOCAB)} candidates -> {a.out} ({time.time()-t0:.0f}s)")

if __name__=="__main__":
    main()
