"""Driving-decision questions from a future ego trajectory (labels are anchored on the realised future).
Features of a 5 s ego-frame trajectory + current kinematics -> per-type GBDT fitted on TRAIN questions with
the TRUE future. Evaluated on validation with (a) the true future (ceiling) and (b) a planner's predicted
future (GLOBAL (x, y, yaw) npz: names, traj (N, 51, 3), index 0 = ego at frame 100).
usage: python src/decision_traj.py [--pred runs_jepa/v71_validation.npz [+w ...]]
"""
import argparse, glob, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "recon", "devkit"))
from nureasoning.common.pickle_io import load_pickle
SUBS = ("driving_decision_longitudinal", "driving_decision_lateral", "driving_decision_joint")


def ego_frame(G, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2]); d = G[:, :2] - pose[:2]
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], np.arctan2(np.sin(G[:, 2] - pose[2]), np.cos(G[:, 2] - pose[2]))], 1)


def feats(E, v0, a0):
    """E: (51, 3) ego-frame poses at 10 Hz, E[0] = origin."""
    seg = np.hypot(*np.diff(E[:, :2], axis=0).T) / 0.1
    f = {"v0": v0, "a0": a0}
    for t in (5, 10, 20, 30, 50):
        f[f"v{t}"] = float(seg[max(0, t - 5):t].mean()); f[f"x{t}"] = float(E[t, 0]); f[f"y{t}"] = float(E[t, 1]); f[f"h{t}"] = float(E[t, 2])
    f["vmin"] = float(seg.min()); f["vmax"] = float(seg.max()); f["dv10"] = f["v10"] - v0; f["dv30"] = f["v30"] - v0
    f["ymax"] = float(E[:, 1].max()); f["ymin"] = float(E[:, 1].min()); f["dist"] = float(seg.sum() * 0.1)
    return f


def clip_rows(split, clip, Gfut=None):
    meta = json.load(open(f"cache/{split}/{clip}/metadata.json"))
    e = load_pickle(f"cache/{split}/{clip}/{meta['frames'][100]['ego_state']}")
    pose = np.array([e.pose["x"], e.pose["y"], e.pose["yaw"]])
    v0 = math.hypot(e.velocity.get("vx", 0), e.velocity.get("vy", 0)); a0 = e.acceleration.get("ax", 0.0)
    if Gfut is None:
        T = np.array(e.trajectory_future, float)
        if len(T) < 50: return None
        Gfut = np.vstack([pose[None], T[:50]])
    return feats(ego_frame(np.asarray(Gfut, float), pose), v0, a0)


def questions(split):
    root = {"train": "vqa_train", "validation": "vqa_val"}[split]; out = []
    for f in glob.glob(f"{root}/*/*_vqa.json"):
        J = json.load(open(f))
        for q in J["questions"]:
            if q["subcategory"] in SUBS: out.append((f.split("/")[1], q))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pred", default=""); a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    cache = {}
    def F(split, clip, G=None):
        k = (split, clip, G is not None)
        if k not in cache: cache[k] = clip_rows(split, clip, G)
        return cache[k]
    tr = questions("train"); va = questions("validation")
    pred = {}
    if a.pred:
        D = []
        for part in a.pred.split("+"):
            path, _, w = part.partition("*"); D.append((np.load(path), float(w) if w else 1.0))
        names = list(D[0][0]["names"])
        for i, n in enumerate(names):
            Gs = [d["traj"][list(d["names"]).index(n)] for d, _ in D if n in set(d["names"])]
            ws = np.array([w for (d, w) in D if n in set(d["names"])]); ws = ws / ws.sum()
            g = sum(w * G for w, G in zip(ws, Gs))
            g[:, 2] = np.arctan2(sum(w * np.sin(G[:, 2]) for w, G in zip(ws, Gs)), sum(w * np.cos(G[:, 2]) for w, G in zip(ws, Gs)))
            pred[n] = g
    for sub in SUBS:
        Xtr, ytr = [], []
        for c, q in tr:
            if q["subcategory"] != sub: continue
            f = F("train", c)
            if f: Xtr.append(list(f.values())); ytr.append(q["choices"][q["answer"]])
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, l2_regularization=1.0).fit(np.array(Xtr), ytr)
        def acc(G=False):
            ok = n = 0
            for c, q in va:
                if q["subcategory"] != sub: continue
                f = F("validation", c, pred.get(c)) if G else F("validation", c)
                if f is None or (G and c not in pred): continue
                p = dict(zip(m.classes_, m.predict_proba(np.array([list(f.values())]))[0]))
                best = max(q["choices"], key=lambda k: p.get(q["choices"][k], 0.0)); ok += best == q["answer"]; n += 1
            return ok / max(n, 1), n
        line = f"{sub:30s} train n={len(ytr)} | val TRUE future {acc()[0]:.3f}"
        if pred: r = acc(True); line += f" | val PREDICTED future {r[0]:.3f} (n={r[1]})"
        print(line, flush=True)


if __name__ == "__main__":
    main()
