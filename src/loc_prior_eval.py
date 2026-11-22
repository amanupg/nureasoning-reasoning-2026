"""Location speed prior check: does the human 5 s distance of TRAIN clips that passed the same spot with the
same heading predict a validation clip's 5 s distance beyond its own ego history?
Neighbours from train only (never validation/test futures). Validation = held-out score.
  .venv/bin/python src/loc_prior_eval.py
"""
import json, os
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
A = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
P = {s: json.load(open(f"{A}/loc_pose_{s}.json")) for s in ["train", "validation", "test"]}
M = {s: json.load(open(f"{A}/meta_labels_{s}.json")) for s in ["train", "validation", "test"]}
CMDS = sorted({str(o.get("command")) for s in M for o in M[s].values()})


def hist(o):
    return [o.get(f"v_m{k}") if o.get(f"v_m{k}") is not None else np.nan for k in range(6)] + \
           [o["ax"], o["yaw_rate"] if o["yaw_rate"] is not None else np.nan, CMDS.index(str(o.get("command")))]


tr = [c for c in P["train"] if c in M["train"] and "dist5" in M["train"][c]]
XY = np.array([[P["train"][c]["x"], P["train"][c]["y"]] for c in tr]); YAW = np.array([P["train"][c]["yaw"] for c in tr])
D5 = np.array([M["train"][c]["dist5"] for c in tr]); V0 = np.array([M["train"][c]["v_m0"] for c in tr])
LOG = np.array([P["train"][c]["log"] for c in tr]); tree = cKDTree(XY)


def loc_feats(split, c, R, excl_log=None):
    p = P[split][c]; idx = tree.query_ball_point([p["x"], p["y"]], R)
    idx = [i for i in idx if abs((YAW[i] - p["yaw"] + np.pi) % (2 * np.pi) - np.pi) < np.pi / 6 and LOG[i] != excl_log]
    if not idx: return [0, np.nan, np.nan, np.nan, np.nan]
    d = D5[idx]; return [len(idx), np.median(d), np.percentile(d, 25), np.percentile(d, 75), np.median(V0[idx])]


for R in [10, 20, 40]:
    rows = {}
    for s in ["train", "validation", "test"]:
        cl = [c for c in P[s] if c in M[s] and (s == "test" or "dist5" in M[s][c])]
        # train rows: leave own log out so the prior never contains the clip's own future
        L = np.array([loc_feats(s, c, R, P[s][c]["log"] if s == "train" else None) for c in cl])
        H = np.array([hist(M[s][c]) for c in cl], float)
        y = np.array([M[s][c].get("dist5", np.nan) for c in cl])
        rows[s] = (cl, H, L, y)
    cov = {s: float((rows[s][2][:, 0] > 0).mean()) for s in rows}
    out = {}
    for name, f in [("history", lambda H, L: H), ("history+loc", lambda H, L: np.c_[H, L])]:
        ms = []
        for seed in range(3):
            g = HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, learning_rate=0.05, random_state=seed)
            g.fit(f(rows["train"][1], rows["train"][2]), rows["train"][3])
            ms.append(g.predict(f(rows["validation"][1], rows["validation"][2])))
        out[name] = np.mean(ms, 0)
    yv = rows["validation"][3]; cv = rows["validation"][2][:, 0] > 0
    e0 = np.abs(out["history"] - yv); e1 = np.abs(out["history+loc"] - yv)
    dd = e0 - e1; bs = [dd[np.random.default_rng(b).integers(0, len(dd), len(dd))].mean() for b in range(2000)]
    print(f"R={R} m coverage train/val/test {cov['train']:.2f}/{cov['validation']:.2f}/{cov['test']:.2f} | val dist5 MAE "
          f"history {e0.mean():.3f} -> +loc {e1.mean():.3f} (gain {dd.mean():.3f} m, 95% CI {np.percentile(bs,2.5):.3f}..{np.percentile(bs,97.5):.3f}); "
          f"covered n={cv.sum()}: {e0[cv].mean():.3f} -> {e1[cv].mean():.3f}", flush=True)
    if R == 20:
        json.dump({s: {c: dict(zip(["n", "d5_med", "d5_q25", "d5_q75", "v0_med"], [None if (isinstance(v, float) and np.isnan(v)) else float(v) for v in l]))
                       for c, l in zip(rows[s][0], rows[s][2])} for s in rows},
                  open(f"{A}/loc_prior_r20.json", "w"))
