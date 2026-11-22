"""Evaluate correspondence motion features on future_motion_label (validation).
Diagnostic vs annotated velocity, then accuracy of a classifier on held-out logs."""
import json, os, random, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reason_model import load, norm_text
from sklearn.ensemble import HistGradientBoostingClassifier as HGB

MF = ["du", "dv", "s_best", "s_zero", "vx", "vy", "v", "v_radial", "k_best", "v_away_scale", "s_k0.8", "s_k0.9", "s_k1.12", "s_k1.25"]


def ann_vel(split, clip, tok, _c={}):
    if clip not in _c:
        fr = json.load(open(f"cache/{split}/{clip}/metadata.json"))["frames"]; d = {}
        for k in (100, 101, 99):
            r = fr[k].get("reasoning")
            if r and os.path.isfile(f"cache/{split}/{clip}/{r}"):
                J = json.load(open(f"cache/{split}/{clip}/{r}"))
                for t, o in (J.get("Spatial", {}).get("cross_view_correspondence") or {}).items():
                    d[t] = (o["velocity_3d_ego"]["x"], o["velocity_3d_ego"]["y"])
                break
        _c.clear(); _c[clip] = d
    return _c[clip].get(tok)


def rows(qs, M, cf, split):
    X, Y, G, V = [], [], [], []
    for q in qs:
        m = M.get(q["question_id"]); c = cf.get(q["_clip"], {}); f = q["_f"]
        if m is None: continue
        X.append([f.get("x", np.nan), f.get("y", np.nan), c.get("c_v", np.nan), c.get("c_ax", np.nan)] +
                 [m.get(k, np.nan) for k in MF])
        Y.append(norm_text(q["answer_text"])); G.append("_".join(q["_clip"].split("_")[:2]))
        V.append(ann_vel(split, q["_clip"], q.get("target_track_token")) or (np.nan, np.nan))
    return np.array(X, float), np.array(Y), np.array(G), np.array(V, float)


if __name__ == "__main__":
    qs = [q for q in load() if q.get("subcategory") == "future_motion_label"]
    M = json.load(open("artifacts/motion_validation.json")); cf = json.load(open("artifacts/clipfeat_validation.json"))
    X, Y, G, V = rows(qs, M, cf, "validation")
    ok = ~np.isnan(X[:, 8]) & ~np.isnan(V[:, 0])
    for i, nm in ((8, "vx"), (9, "vy")):
        print(f"  corr(est {nm}, annotated) = {np.corrcoef(X[ok, i], V[ok, i - 8])[0, 1]:.3f}   median |err| {np.median(np.abs(X[ok, i] - V[ok, i - 8])):.2f} m/s")
    sp = np.hypot(V[ok, 0], V[ok, 1]) > 1.0
    print(f"  moving (>1 m/s) vs stopped: est speed median {np.median(X[ok][sp, 10]):.2f} vs {np.median(X[ok][~sp, 10]):.2f}")
    logs = sorted(set(G)); res = []
    for seed in range(5):
        random.Random(seed).shuffle(logs); h = set(logs[:len(logs) // 2]); fit = np.array([g in h for g in G])
        a0 = np.mean(HGB(max_iter=300, random_state=0).fit(X[fit][:, :4], Y[fit]).predict(X[~fit][:, :4]) == Y[~fit])
        a1 = np.mean(HGB(max_iter=300, random_state=0).fit(X[fit], Y[fit]).predict(X[~fit]) == Y[~fit])
        res.append((a0, a1))
    r = np.array(res)
    print(f"held-out (val half->half, 5 splits): base {r[:, 0].mean():.3f}   + motion {r[:, 1].mean():.3f}   [oracle velocity 0.809]")
