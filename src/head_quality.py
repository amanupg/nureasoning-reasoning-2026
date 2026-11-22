"""Can the sub-metric heads predict anything at all from ego+route features?

If the collision head is near chance, the vocabulary oracle headroom is real but
UNREACHABLE without perception, and no amount of selection-weight tuning helps.
This is the decisive diagnostic before investing in the scorer.
"""
import json, os, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scorer import build_xy, usable_cols, SUBS

data = json.load(open("artifacts/vocab_labels_val.json"))
X, Y, ci, K, vocab = build_xy(data)
n = ci.max() + 1
rng = np.random.RandomState(0); perm = rng.permutation(n)
tr = np.isin(ci, perm[:int(n*0.7)]); ev = np.isin(ci, perm[int(n*0.7):])
cols = usable_cols(X[tr])

print(f"train rows {tr.sum()}  eval rows {ev.sum()}  features {len(cols)}")
print(f"\n{'sub-metric':10} {'base rate / mean':>18} {'metric':>10} {'value':>8}  {'vs chance':>10}")
for i, name in enumerate(SUBS):
    y = Y[:, i]
    if name in ("col", "da", "comf"):
        b = (y > 0.99).astype(int)                 # "perfect score" vs not
        m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, random_state=0)
        m.fit(X[tr][:, cols], b[tr])
        if len(np.unique(b[ev])) < 2:
            print(f"{name:10} {b.mean():>18.3f} {'AUC':>10} {'n/a':>8}  (single class)")
            continue
        p = m.predict_proba(X[ev][:, cols])[:, 1]
        auc = roc_auc_score(b[ev], p)
        print(f"{name:10} {b.mean():>18.3f} {'AUC':>10} {auc:>8.3f}  {auc-0.5:>+10.3f}")
    else:
        m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, random_state=0)
        m.fit(X[tr][:, cols], y[tr])
        pr = m.predict(X[ev][:, cols])
        r = np.corrcoef(pr, y[ev])[0, 1]
        base = np.full_like(y[ev], y[tr].mean())
        r2 = 1 - ((y[ev]-pr)**2).sum() / ((y[ev]-base)**2).sum()
        print(f"{name:10} {y.mean():>18.3f} {'R2':>10} {r2:>8.3f}  corr {r:>+.3f}")
