"""Within-clip ranking quality - the property selection actually needs.

A head can score a high global AUC purely by knowing which CLIPS are dangerous,
while being unable to say which CANDIDATE is safer inside a given clip. Only the
latter helps a selector. This measures the second thing.
"""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scorer import build_xy, usable_cols, SUBS, true_nps

data = json.load(open("artifacts/vocab_labels_val.json"))
X, Y, ci, K, vocab = build_xy(data)
n = ci.max() + 1
rng = np.random.RandomState(0); perm = rng.permutation(n)
tr = np.isin(ci, perm[:int(n*0.7)]); ev = np.isin(ci, perm[int(n*0.7):])
cols = usable_cols(X[tr])

preds = {}
for i, name in enumerate(SUBS):
    y = Y[:, i]
    if name in ("col", "da", "comf"):
        m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, random_state=0)
        m.fit(X[tr][:, cols], np.round(y[tr]*2).astype(int))
        p = m.predict_proba(X[ev][:, cols])
        preds[name] = (p * (m.classes_[None, :]/2.0)).sum(axis=1)
    else:
        m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, random_state=0)
        m.fit(X[tr][:, cols], y[tr])
        preds[name] = np.clip(m.predict(X[ev][:, cols]), 0, 1)

Ye = Y[ev].reshape(-1, K, len(SUBS))
print(f"eval clips {len(Ye)}   vocabulary {K}\n")
print(f"{'sub-metric':10} {'within-clip Spearman':>22} {'clips with any variation':>26}")
for i, name in enumerate(SUBS):
    P = preds[name].reshape(-1, K)
    rhos, varied = [], 0
    for c in range(len(Ye)):
        t = Ye[c, :, i]
        if np.unique(t).size < 2:
            continue
        varied += 1
        r = spearmanr(P[c], t).statistic
        if r == r: rhos.append(r)
    print(f"{name:10} {np.mean(rhos):>22.3f} {100*varied/len(Ye):>25.1f}%")

# and the thing that matters: ranking the true NPS
NPS = true_nps(Y[ev], K)
S = (np.log(preds["col"].reshape(-1,K)+1e-6) + np.log(preds["da"].reshape(-1,K)+1e-6)
     + np.log(0.3*preds["prog"].reshape(-1,K) + 0.2*preds["comf"].reshape(-1,K)
              + 0.5*preds["hum"].reshape(-1,K) + 1e-6))
rhos=[spearmanr(S[c], NPS[c]).statistic for c in range(len(NPS)) if np.unique(NPS[c]).size>1]
rhos=[r for r in rhos if r==r]
print(f"\n{'FULL NPS':10} {np.mean(rhos):>22.3f}  (this is what selection ranks on)")
cur = int(np.where([v == (1.0, 0.0) for v in map(tuple, vocab)])[0][0])
print(f"\n  current planner  {NPS[:,cur].mean():.4f}")
print(f"  argmax predicted {NPS[np.arange(len(NPS)), S.argmax(1)].mean():.4f}")
print(f"  oracle           {NPS.max(1).mean():.4f}")
print(f"  random candidate {NPS.mean():.4f}")
