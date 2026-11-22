"""distance_from_2d_bbox by motion parallax (static objects): the ego's own known motion over the last
second triangulates the object.

Inputs: fm_track.py tracks (--sub distance_from_2d_bbox): per frame 90..100 the tracked mask box and the
camera timestamp. Ego poses (full 6-DoF, world) interpolated to each camera timestamp; camera extrinsics
from the clip calibration (camgeo: lidar frame = ego frame).
Pose time: ego pose looked up at camera time + POSE_LAG (one frame; see LEDGER).
Model (static object): X lies on the frame-100 ray through the STATED box centre at range s. For each
earlier frame t: bearing residual = angle between the camera-t ray through the tracked box centre and
the direction to X; scale residual = log(z_t(X) * h_t) - log(z_100(X) * h_100) (pinhole: depth x box
height is constant for a rigid object). 1-D search over s minimising
  sum(bearing / SB)^2 + sum(scale / SS)^2.
Outputs per question: estimate (ego-frame distance at frame 100 = hypot(x, y)), normalised residual,
baseline (camera travel over the window), number of frames.
usage: python src/dist_parallax.py <split> <track_prefix> [--truth]
"""
import argparse, glob, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "recon", "devkit")]
import camgeo
from dist_mono import pick
from nureasoning.common.pickle_io import load_pickle
SB, SS = math.radians(0.25), float(os.environ.get("PLX_SS", 0.04))
POSE_LAG = 0.1      # s: ego poses lead the camera timestamps by one frame (validation, two log halves agree: 0.08-0.10)


def quat(qw, qx, qy, qz):
    return camgeo._quat(qw, qx, qy, qz)


class Poses:
    def __init__(self, split, clip, meta):
        self.ts, self.T, self.Q = [], [], []
        for k in range(85, 101):
            f = meta["frames"][k]; e = load_pickle(f"cache/{split}/{clip}/{f['ego_state']}").pose
            q = np.array([e["qw"], e["qx"], e["qy"], e["qz"]])
            if self.Q and np.dot(q, self.Q[-1]) < 0: q = -q
            self.ts.append(f["timestamp_us"] / 1e6); self.T.append(np.array([e["x"], e["y"], e["z"]])); self.Q.append(q)
        self.ts = np.array(self.ts)

    def at(self, t):
        t = t + POSE_LAG
        i = int(np.clip(np.searchsorted(self.ts, t) - 1, 0, len(self.ts) - 2)); a = (t - self.ts[i]) / (self.ts[i + 1] - self.ts[i])
        a = float(np.clip(a, -0.5, 1.5))
        T = (1 - a) * self.T[i] + a * self.T[i + 1]; q = (1 - a) * self.Q[i] + a * self.Q[i + 1]
        return quat(*(q / np.linalg.norm(q))), T


def cam_ray(c, u, v):
    pc = np.linalg.inv(c["K"]) @ np.array([u / 1000 * c["W"], v / 1000 * c["H"], 1.0])
    return pc / np.linalg.norm(pc)


def solve(rec, rig, poses, edge=3.0):
    c = rig.cams.get(rec["cam"])
    if c is None or rec.get("missing") or rec["box"][0] is None: return None
    Rw0, Tw0 = poses.at(rec["t"][0])                               # ego->world at the frame-100 camera time
    Rc0, Cc0 = Rw0 @ c["R"], Rw0 @ c["t"] + Tw0                    # camera->world
    bq = rec["box_q"]; r0 = Rc0 @ cam_ray(c, (bq[0] + bq[2]) / 2, (bq[1] + bq[3]) / 2)
    h0 = rec["box"][0][3] - rec["box"][0][1]
    obs = []
    for i in range(1, len(rec["frames"])):
        b = rec["box"][i]
        if b is None or (rec["score"][i] is not None and rec["score"][i] < 0): continue
        if min(b[0], b[1]) < edge or max(b[2], b[3]) > 1000 - edge: continue
        Rw, Tw = poses.at(rec["t"][i]); Rc, Cc = Rw @ c["R"], Rw @ c["t"] + Tw
        obs.append((Rc, Cc, Rc @ cam_ray(c, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2), b[3] - b[1]))
    if len(obs) < 3: return None
    base = max(np.linalg.norm(o[1] - Cc0) for o in obs)

    def cost(s):
        X = Cc0 + s * r0; z0 = (Rc0.T @ (X - Cc0))[2]; tot = 0.0
        for Rc, Cc, ray, h in obs:
            d = X - Cc; n = np.linalg.norm(d)
            ang = math.acos(float(np.clip(d @ ray / n, -1, 1)))
            zt = (Rc.T @ d)[2]
            sc = math.log(max(zt, 1e-3) * h) - math.log(max(z0, 1e-3) * h0)
            tot += (ang / SB) ** 2 + (sc / SS) ** 2
        return tot / len(obs)
    grid = np.exp(np.linspace(math.log(1.0), math.log(150.0), 400))
    cs = [cost(s) for s in grid]; j = int(np.argmin(cs))
    lo, hi = grid[max(0, j - 1)], grid[min(len(grid) - 1, j + 1)]
    for _ in range(40):                                             # golden-section refine
        m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        if cost(m1) < cost(m2): hi = m2
        else: lo = m1
    s = (lo + hi) / 2; X = Cc0 + s * r0
    Pe = Rw0.T @ (X - Tw0)                                          # ego frame at frame 100 (camera time)
    # curvature: how sharply the cost rises around the optimum (well-posed vs flat)
    sharp = (cost(s * 1.05) + cost(s / 1.05)) / 2 - cost(s)
    return dict(est=float(math.hypot(Pe[0], Pe[1])), x=float(Pe[0]), y=float(Pe[1]), res=float(cost(s)), sharp=float(sharp), base=float(base), n=len(obs))


def truth_static(split, recs):
    def walk(o):
        if isinstance(o, dict):
            if "track_token" in o and "velocity_3d_ego" in o: yield o
            for v in o.values(): yield from walk(v)
        elif isinstance(o, list):
            for v in o: yield from walk(v)
    Q = {}
    for f in glob.glob("vqa_val/*/*_vqa.json"):
        for q in json.load(open(f)).get("questions") or []: Q[q["question_id"]] = q.get("target_track_token")
    spd, cache = {}, {}
    for qid, r in recs.items():
        if r["clip"] not in cache:
            cache[r["clip"]] = {o["track_token"]: o for fn in glob.glob(f"cache/{split}/{r['clip']}/reasoning/*.json")
                                for R in [json.load(open(fn))] if abs(R["frame_index"] - 100) <= 1 for o in walk(R)}
        o = cache[r["clip"]].get(Q.get(qid))
        if o: spd[qid] = math.hypot(o["velocity_3d_ego"]["x"], o["velocity_3d_ego"]["y"])
    return spd


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("split"); ap.add_argument("prefix"); ap.add_argument("--truth", action="store_true")
    a = ap.parse_args()
    recs = {}
    for f in sorted(glob.glob(f"{a.prefix}_s*of*.json")): recs.update(json.load(open(f)))
    rigs, poses, out = {}, {}, {}
    for qid, r in recs.items():
        if r.get("missing"): continue
        cl = r["clip"]
        if cl not in rigs:
            meta = json.load(open(f"cache/{a.split}/{cl}/metadata.json")); rigs[cl] = camgeo.Rig(meta); poses[cl] = Poses(a.split, cl, meta)
        s = solve(r, rigs[cl], poses[cl])
        if s: out[qid] = s
    json.dump(out, open(f"{a.prefix}_parallax.json", "w"))
    print(f"{len(recs)} tracked, solved {len(out)}")
    if not a.truth: return
    spd = truth_static(a.split, recs)
    G = {json.loads(l)["question_id"]: json.loads(l) for l in open("artifacts/gbdt_heldout_preds.jsonl")}
    acc = lambda qs: np.mean([pick(out[q]["est"], recs[q]["choices"]) == recs[q]["answer"] for q in qs]) if qs else float("nan")
    tv = lambda q: float(recs[q]["choices"][recs[q]["answer"]].split()[0])
    for name, sel in (("all solved", lambda q: True), ("truly static (<0.3 m/s)", lambda q: spd.get(q, 9) < 0.3),
                      ("truly moving", lambda q: spd.get(q, 0) >= 0.3)):
        qs = [q for q in out if sel(q)]
        if qs: print(f"{name:26s} n={len(qs):4d} acc {acc(qs):.3f} med|log err| {np.median([abs(math.log(out[q]['est'] / tv(q))) for q in qs]):.3f}")
    print("gates (observable only): baseline >= b, residual <= r  -> n, parallax acc, GBDT acc on the same (held-out subset)")
    for b in (0.5, 1.0, 2.0, 4.0):
        for rr in (1.0, 3.0, 10.0, 1e9):
            qs = [q for q in out if out[q]["base"] >= b and out[q]["res"] <= rr]
            qg = [q for q in qs if q in G]
            if len(qs) < 20: continue
            print(f"  b>={b:3.1f} r<={rr:6.0f}: n={len(qs):4d} acc {acc(qs):.3f} | held-out n={len(qg)} parallax {acc(qg):.3f} GBDT {np.mean([G[q]['model'] == G[q]['gold'] for q in qg]):.3f}")


if __name__ == "__main__":
    main()
