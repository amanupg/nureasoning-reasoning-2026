"""GLOBAL (x, y, yaw) 51-pose trajectories of a chooser recipe for any split, for blending by other
sessions (format of ../jepa-nureasoning/runs/*.npz: names, traj (N, 51, 3), index 0 = ego at frame 100).

Same logic as make_submission.plan(): a source pick (k >= 148) is built from its block exactly as
labelled (label_paths SPECS: re-timed or raw, mean of the model's samples); route picks -> vocab148;
no pick or untrusted route without a source pick -> mean of the FALLBACK sources' paths (circular yaw).
Source paths for the split come from the registry's test path with "/test" -> "/<split>".

usage: python src/build_traj.py <split> <selections.json> <out.npz> [--fallback a15ft_h0+a15ft_h1]
       python src/build_traj.py <split> --raw <registry-name> <out.npz>     (the model's raw mean path)
Parity: for validation, source picks are compared with the stored label trajectories.
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmean(G):
    g = np.mean(G, 0)
    g[:, 2] = np.arctan2(np.mean([np.sin(x[:, 2]) for x in G], 0), np.mean([np.cos(x[:, 2]) for x in G], 0))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("sel", nargs="?"); ap.add_argument("out", nargs="?")
    ap.add_argument("--raw", default=""); ap.add_argument("--fallback", default="a15ft_h0+a15ft_h1")
    a = ap.parse_args()
    if a.raw:
        a.out = a.sel
    import make_submission as M, planners
    from vocab148 import VOCAB148, build
    from label_paths import SPECS as SP28
    from label_a15 import SPECS as SP27
    from a15_calib import mean_global, retime
    reg = json.load(open("artifacts/path_sources.json"))
    path = lambda name: reg[name]["test"].replace("/test", f"/{a.split}")

    def src_path(name, clip):
        s = reg[name]
        if s["kind"] == "a15":
            return mean_global(np.load(os.path.join(path(name), clip + ".npz")))
        d = np.load(path(name)); return np.asarray(d["traj"][list(d["names"]).index(clip)], float)

    S = json.load(open(a.sel)) if not a.raw else None
    names, trajs, kinds = [], [], []
    for c in sorted(os.listdir(f"cache/{a.split}")):
        cp = f"cache/{a.split}/{c}"
        if not os.path.isfile(f"{cp}/metadata.json"): continue
        meta = M._load_metadata(cp); fr = meta["frames"]; ki = M.select_key_frame_idx(fr, M.CFG.key_frame_index)
        ego = M.load_pickle(os.path.join(cp, fr[ki]["ego_state"])); route = M.build_route_line(fr[ki].get("mission_goal"))
        if a.raw:
            g = src_path(a.raw, c); kinds.append("raw")
        else:
            k = S["selections"].get(c); trust = M.route_trustworthy(ego, route)
            if k is not None and k >= 148:
                off = 0
                for name, labdir, K in S["blocks"]:
                    if k < off + K: break
                    off += K
                j = k - off; sname = labdir.replace("labels_", "", 1); specs = SP27 if reg[sname].get("specs") == 27 else SP28
                g = src_path(sname, c); kind, v = specs[j]
                if kind != "raw":
                    v0 = planners._speed(ego); kin = planners.arc_length_kinematic(v0, planners._lon_accel(ego))
                    own = float(np.sum(np.hypot(*np.diff(g[:, :2], axis=0).T)))
                    L = kin * v if kind == "rel" else (v if kind == "abs" else own)
                    g = retime(g, v0, max(0.0, L), ego.pose["yaw"])
                kinds.append(sname)
            elif not trust or k is None:
                g = cmean([src_path(x, c) for x in a.fallback.split("+")]); kinds.append("fallback")
            else:
                g = build(meta, fr, ki, ego, route, VOCAB148[k]); kinds.append("route")
        names.append(c); trajs.append(np.asarray(g, float))
    np.savez_compressed(a.out, names=np.array(names), traj=np.array(trajs), kind=np.array(kinds))
    from collections import Counter
    print(f"{a.split}: {len(names)} -> {a.out} {dict(Counter(kinds))}")


if __name__ == "__main__":
    main()
