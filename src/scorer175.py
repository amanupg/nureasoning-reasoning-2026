"""Chooser over 148 route candidates + 27 Alpamayo-path candidates (175).

Same per-sub-metric GBDT heads and log-linear selection as scorer148_novision, on
clips that have both label sets. Extra features: is_alpamayo flag, Alpamayo
clip features (sample spread, own arc, kinematic arc) for every candidate.
Train fit/calib split = seed-0 85/15 on train clips; evaluated on validation.
Writes test selections as indices into the 175 list (0..147 = VOCAB148 order,
148..174 = label_a15.SPECS order).
"""
import glob, json, os, sys, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def load(split):
    A = [np.load(f) for f in sorted(glob.glob(f"artifacts/labels148/{split}_[0-9]*.npz"))]
    B = [np.load(f) for f in sorted(glob.glob(f"artifacts/labels_a15/{split}_[0-9]*.npz"))]
    a = {k: np.concatenate([p[k] for p in A]) for k in A[0].files if k != "vocab"}
    b = {k: np.concatenate([p[k] for p in B]) for k in B[0].files if k != "vocab"}
    bi = {n: i for i, n in enumerate(b["names"])}
    keep = [i for i, n in enumerate(a["names"]) if n in bi]; j = [bi[a["names"][i]] for i in keep]
    va = np.concatenate([A[0]["vocab"], np.zeros((148, 1), np.float32)], 1)
    vb = np.concatenate([B[0]["vocab"], np.ones((len(B[0]["vocab"]), 1), np.float32)], 1)
    d = {"names": a["names"][keep], "feat": a["feat"][keep], "a15feat": b["a15feat"][j],
         "traj": np.concatenate([a["traj"][keep], b["traj"][j]], 1), "vocab": np.concatenate([va, vb])}
    if "labels" in a:
        d["labels"] = np.concatenate([a["labels"][keep], b["labels"][j]], 1)
    return d


def rows(d):
    n, K = d["traj"].shape[:2]; tr = d["traj"].astype(np.float32)
    X = np.concatenate([np.repeat(d["feat"][:, None, :], K, 1), np.repeat(d["a15feat"][:, None, :], K, 1),
                        np.repeat(d["vocab"][None], n, 0), tr[:, :, -1, :], tr[:, :, 5, :]], -1).reshape(n * K, -1)
    return X, n, K


nps = lambda y: y[..., 0] * y[..., 1] * (0.3 * y[..., 2] + 0.2 * y[..., 3] + 0.5 * y[..., 4])


def main():
    only148 = os.environ.get("ONLY148") == "1"
    tr, va = load("train"), load("validation")
    if only148:
        for d in (tr, va):
            d["traj"] = d["traj"][:, :148]; d["labels"] = d["labels"][:, :148]; d["vocab"] = d["vocab"][:148]
    Xt, nt, K = rows(tr); Xv, nv, _ = rows(va)
    Yt = tr["labels"].reshape(nt * K, 5); Yv = va["labels"]
    perm = np.random.RandomState(0).permutation(nt); fit_c, cal_c = perm[:int(.85 * nt)], perm[int(.85 * nt):]
    m_fit = np.isin(np.repeat(np.arange(nt), K), fit_c); m_cal = ~m_fit
    print(f"train {nt} clips (fit {len(fit_c)} / calib {len(cal_c)}), validation {nv}, K={K}", flush=True)
    H = {}; t0 = time.time()
    for i, name in enumerate(["col", "da", "prog", "comf", "hum"]):
        y = Yt[m_fit, i]
        if name in ("col", "da", "comf"):
            m = HistGradientBoostingClassifier(max_iter=400, learning_rate=.06, max_leaf_nodes=63, random_state=0)
            m.fit(Xt[m_fit], np.round(y * 2).astype(int))
        else:
            m = HistGradientBoostingRegressor(max_iter=400, learning_rate=.06, max_leaf_nodes=63, random_state=0)
            m.fit(Xt[m_fit], y)
        H[name] = m
    print(f"heads fitted in {time.time()-t0:.0f}s", flush=True)

    def pred(X, n):
        o = {}
        for k, m in H.items():
            if k in ("col", "da", "comf"):
                p = m.predict_proba(X); o[k] = (p * (m.classes_[None] / 2.0)).sum(1).reshape(n, K)
            else:
                o[k] = np.clip(m.predict(X), 0, 1).reshape(n, K)
        return o
    S = lambda p, w: w[0] * np.log(p["col"] + 1e-6) + w[1] * np.log(p["da"] + 1e-6) + \
        np.log(.3 * p["prog"] + .2 * p["comf"] + .5 * p["hum"] + 1e-6)
    pc = pred(Xt[m_cal], len(cal_c)); Nc = nps(Yt[m_cal].reshape(len(cal_c), K, 5))
    best = max(((Nc[np.arange(len(Nc)), S(pc, (a, b)).argmax(1)].mean(), (a, b))
                for a in (0, .25, .5, 1, 2, 4, 8) for b in (0, .25, .5, 1, 2, 4, 8)))
    pv = pred(Xv, nv); Nv = nps(Yv); sel = S(pv, best[1]).argmax(1)
    print(f"weights {best[1]} (calib {best[0]:.4f})")
    print(f"VALIDATION {nv} clips: default {Nv[:, 38].mean():.4f}  selected {Nv[np.arange(nv), sel].mean():.4f}  "
          f"oracle {Nv.max(1).mean():.4f}  picks Alpamayo {np.mean(sel >= 148):.2f}", flush=True)
    if only148: return
    te = load("test"); Xe, ne, _ = rows(te)
    pick = S(pred(Xe, ne), best[1]).argmax(1)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = f"artifacts/versioned/selections_test_scorer175_{stamp}.json"
    json.dump({"split": "test", "cands": "labels148+labels_a15", "model": "gbdt_175", "w": best[1],
               "val_selected": float(Nv[np.arange(nv), sel].mean()), "val_default": float(Nv[:, 38].mean()),
               "selections": {str(n): int(k) for n, k in zip(te["names"], pick)},
               "alpamayo_share": float(np.mean(pick >= 148))}, open(out, "w"))
    print(f"TEST {ne} clips, Alpamayo picked {np.mean(pick >= 148):.2f} -> {out}")


if __name__ == "__main__":
    main()
