"""Fixing argmax overoptimisation over a noisy scorer.

Plain argmax over 63 candidates loses to the default choice even though the
scorer ranks at Spearman 0.478, because argmax selects whichever candidate got
a lucky PREDICTION. Three standard remedies, measured:

  1. Empirical-Bayes shrinkage toward the per-candidate prior (its mean NPS on
     training clips), which is the single most informative thing we know a priori.
  2. Trust region: only consider candidates near the default.
  3. Top-k mean: average the true outcome over the k best-predicted candidates
     (a diagnostic, not deployable) to show how fast the curse decays with k.
"""
import json, os, sys
import numpy as np
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

NPS_tr = true_nps(Y[tr], K); NPS_ev = true_nps(Y[ev], K)
prior = NPS_tr.mean(axis=0)                       # per-candidate prior from TRAIN clips
S = (np.log(preds["col"].reshape(-1,K)+1e-6) + np.log(preds["da"].reshape(-1,K)+1e-6)
     + np.log(0.3*preds["prog"].reshape(-1,K) + 0.2*preds["comf"].reshape(-1,K)
              + 0.5*preds["hum"].reshape(-1,K) + 1e-6))
Sn = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
Pn = (prior - prior.mean()) / (prior.std() + 1e-9)

cur = int(np.where([v == (1.0, 0.0) for v in map(tuple, vocab)])[0][0])
pick = lambda idx: NPS_ev[np.arange(len(NPS_ev)), idx].mean()
print(f"eval clips {len(NPS_ev)}")
print(f"  default candidate          {pick(np.full(len(NPS_ev), cur)):.4f}")
print(f"  best prior candidate       {pick(np.full(len(NPS_ev), int(prior.argmax()))):.4f}"
      f"   (index {int(prior.argmax())}, vocab {vocab[int(prior.argmax())]})")
print(f"  plain argmax               {pick(S.argmax(1)):.4f}")
print(f"  oracle                     {NPS_ev.max(1).mean():.4f}")

print("\n=== shrinkage toward the per-candidate prior ===")
best=None
for w in np.arange(0.0, 1.01, 0.1):
    v = pick((w*Sn + (1-w)*Pn).argmax(1))
    print(f"  w_model={w:.1f}  NPS={v:.4f}")
    if best is None or v>best[0]: best=(v,w)
print(f"  best: w={best[1]:.1f} -> {best[0]:.4f}")

print("\n=== trust region around the default candidate ===")
sc = np.array([v[0] for v in vocab]); la = np.array([v[1] for v in vocab])
for ds, dl in [(0.15,0.5),(0.3,1.0),(0.45,1.0),(0.6,2.0),(9,9)]:
    ok = (np.abs(sc-1.0)<=ds) & (np.abs(la-0.0)<=dl)
    Sm = np.where(ok[None,:], S, -1e9)
    print(f"  |dscale|<={ds}, |dlat|<={dl}  ({ok.sum():2d} candidates)  NPS={pick(Sm.argmax(1)):.4f}")

print("\n=== how fast does the curse decay with k? (diagnostic) ===")
order = np.argsort(-S, axis=1)
for k in (1,2,3,5,10,63):
    print(f"  mean true NPS of top-{k:2d} predicted: "
          f"{NPS_ev[np.arange(len(NPS_ev))[:,None], order[:,:k]].mean():.4f}")
