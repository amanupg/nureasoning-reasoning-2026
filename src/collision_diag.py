"""Where, when and with what do our at-fault collisions happen?

Human-likeness depends only on the endpoint, so the 49 intermediate waypoints are
free to reshape for the gates. That is only worth doing if we know what we are
hitting and when. This dumps the collision details the harness summary discards.
"""
import os, sys, json, collections
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
CFG=BenchmarkConfig()

def one(cp):
    try:
        meta=_load_metadata(cp); fr=meta.get("frames",[])
        ki=select_key_frame_idx(fr,CFG.key_frame_index)
        ego=load_pickle(os.path.join(cp,fr[ki]["ego_state"]))
        route=build_route_line(fr[ki].get("mission_goal"))
        traj=planners.route_follow(cp,meta,fr,ki,ego,route)
        r=evaluate_clip(cp,CFG,traj)
        if r.error: return None
        d=r.details_collision
        v=ego.velocity if isinstance(ego.velocity,dict) else {}
        return {"clip":r.clip_name,"col":r.s_collision,"nps":r.planning_score,
                "step":d.get("first_at_fault_step",-1),"type":d.get("first_at_fault_type",""),
                "target":d.get("first_at_fault_target",""),
                "n_overlap":d.get("num_overlaps",0),
                "n_atfault":d.get("num_at_fault_overlaps",0),
                "speed":float(np.hypot(v.get("vx",0),v.get("vy",0))),
                "da":r.s_driveable}
    except Exception:
        return None

def main():
    root="cache/validation"
    clips=sorted(os.path.join(root,d) for d in os.listdir(root)
                 if os.path.isfile(os.path.join(root,d,"metadata.json")))
    rows=[]
    with ProcessPoolExecutor() as ex:
        for r in ex.map(one, clips, chunksize=8):
            if r: rows.append(r)
    bad=[r for r in rows if r["col"]<1.0]
    print(f"clips {len(rows)}  at-fault collisions {len(bad)} ({100*len(bad)/len(rows):.1f}%)")
    print("\nby collision TYPE (first at-fault):")
    for k,v in collections.Counter(r["type"] for r in bad).most_common():
        sub=[r for r in bad if r["type"]==k]
        print(f"  {k:26} {v:4d}  target={collections.Counter(r['target'] for r in sub).most_common()}")
    steps=np.array([r["step"] for r in bad if r["step"]>=0])
    print(f"\nWHEN does the first at-fault overlap occur? (step index, 0..50 = 0..5 s)")
    for q in (10,25,50,75,90):
        print(f"  p{q:<3} step {np.percentile(steps,q):5.1f}  = t={np.percentile(steps,q)*0.1:.2f} s")
    print(f"  at step 0 (already overlapping at t=0): {int((steps==0).sum())} clips "
          f"({100*(steps==0).mean():.1f}% of collisions)")
    print(f"  within first 1 s (step<=10): {100*(steps<=10).mean():.1f}%")
    print(f"  after 3 s (step>30)       : {100*(steps>30).mean():.1f}%")
    print("\nego speed at key frame, colliding vs clean:")
    cs=np.array([r["speed"] for r in bad]); gs=np.array([r["speed"] for r in rows if r["col"]==1.0])
    print(f"  colliding p50={np.median(cs):.2f} m/s   clean p50={np.median(gs):.2f} m/s")
    print(f"  stationary (<0.5) share: colliding {100*np.mean(cs<0.5):.1f}%  clean {100*np.mean(gs<0.5):.1f}%")
    json.dump(rows, open("artifacts/collision_diag.json","w"))
    print("\n-> artifacts/collision_diag.json")


if __name__ == "__main__":
    main()
