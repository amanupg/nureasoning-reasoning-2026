"""Scorer trained on TRAIN labels (7,656 clips), evaluated on all of VALIDATION.

Question this answers: with 13x more data, does the ego+route scorer sharpen
enough to beat the default candidate by a meaningful margin? If yes, the v1
failure was data. If it plateaus near +0.01, the gates are not predictable from
ego+route and vision is required.

Selection rules tuned on a held-out 15% of TRAIN clips, never on validation.
"""
import json, os, sys, time, pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scorer import build_xy, usable_cols, SUBS, true_nps

tr = json.load(open("artifacts/vocab_labels_train.json"))
va = json.load(open("artifacts/vocab_labels_val.json"))
Xt, Yt, ct, K, vocab = build_xy(tr)
Xv, Yv, cv, _, _ = build_xy(va)
n = ct.max() + 1
perm = np.random.RandomState(0).permutation(n)
fit = np.isin(ct, perm[:int(n * 0.85)]); cal = np.isin(ct, perm[int(n * 0.85):])
cols = usable_cols(Xt[fit])
print(f"train fit {fit.sum()//K} clips | train calib {cal.sum()//K} | validation {len(Xv)//K}")

t0 = time.time(); heads = {}
for i, name in enumerate(SUBS):
    y = Yt[fit][:, i]
    if name in ("col", "da", "comf"):
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
                                           l2_regularization=1.0, random_state=0)
        m.fit(Xt[fit][:, cols], np.round(y * 2).astype(int)); heads[name] = ("clf", m)
    else:
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
                                          l2_regularization=1.0, random_state=0)
        m.fit(Xt[fit][:, cols], y); heads[name] = ("reg", m)
print(f"fitted heads in {time.time()-t0:.0f}s")


def pred(X):
    out = {}
    for name, (kind, m) in heads.items():
        if kind == "clf":
            p = m.predict_proba(X[:, cols]); out[name] = (p * (m.classes_[None, :] / 2.0)).sum(1)
        else:
            out[name] = np.clip(m.predict(X[:, cols]), 0, 1)
    return {k: v.reshape(-1, K) for k, v in out.items()}


def score(p, w):
    e = 1e-6
    return (w[0] * np.log(p["col"] + e) + w[1] * np.log(p["da"] + e)
            + np.log(0.3 * p["prog"] + 0.2 * p["comf"] + 0.5 * p["hum"] + e))


pc, pv = pred(Xt[cal]), pred(Xv)
Nc, Nv = true_nps(Yt[cal], K), true_nps(Yv, K)
sc = np.array([v[0] for v in vocab]); la = np.array([v[1] for v in vocab])
cur = int(np.where((sc == 1.0) & (la == 0.0))[0][0])
regions = {"all 63": np.ones(K, bool),
           "|ds|<=.3,|dl|<=1": (np.abs(sc - 1) <= .3) & (np.abs(la) <= 1),
           "|ds|<=.45,|dl|<=1": (np.abs(sc - 1) <= .45) & (np.abs(la) <= 1),
           "|ds|<=.6,|dl|<=2": (np.abs(sc - 1) <= .6) & (np.abs(la) <= 2)}
best = None
for rn, ok in regions.items():
    for wn in (0, .25, .5, 1, 2, 4, 8):
        for wd in (0, .25, .5, 1, 2, 4, 8):
            s = np.where(ok[None, :], score(pc, (wn, wd)), -1e9)
            v = Nc[np.arange(len(Nc)), s.argmax(1)].mean()
            if best is None or v > best[0]: best = (v, rn, (wn, wd))
print(f"best on TRAIN-calib: region={best[1]} w_nc={best[2][0]} w_da={best[2][1]} (calib NPS {best[0]:.4f}, "
      f"default {Nc[:, cur].mean():.4f})")

sv = np.where(regions[best[1]][None, :], score(pv, best[2]), -1e9)
sel = sv.argmax(1)
rho = [spearmanr(score(pv, best[2])[c], Nv[c]).statistic for c in range(len(Nv)) if np.unique(Nv[c]).size > 1]
print(f"\nVALIDATION ({len(Nv)} clips), never touched during fitting or tuning:")
print(f"  default candidate         {Nv[:, cur].mean():.4f}")
print(f"  scorer selection          {Nv[np.arange(len(Nv)), sel].mean():.4f}   "
      f"(delta {Nv[np.arange(len(Nv)), sel].mean()-Nv[:, cur].mean():+.4f})")
print(f"  oracle                    {Nv.max(1).mean():.4f}")
print(f"  within-clip Spearman      {np.nanmean(rho):.3f}   (v1 on 594 clips: 0.478)")
print(f"  headroom captured         {100*(Nv[np.arange(len(Nv)), sel].mean()-Nv[:, cur].mean())/(Nv.max(1).mean()-Nv[:, cur].mean()):.1f}%")
pickle.dump({"heads": heads, "cols": cols, "vocab": vocab, "region": best[1], "w": best[2]},
            open(f"artifacts/versioned/scorer_v2_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.pkl", "wb"))
