"""Does the train-only location prior (artifacts/loc_prior_r20.json) explain the planning blend's 5 s length
error e (jepa-nureasoning runs/err_where_validation.json, read-only) on validation? 5-fold CV grouped by log.
Also: date-shift check on inputs only (current speed vs the neighbours' current speed, validation vs test).
"""
import json, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import GroupKFold
E = json.load(open("../nureasoning-planning-2026/runs/err_where_validation.json"))
L = json.load(open("artifacts/loc_prior_r20.json")); P = {s: json.load(open(f"artifacts/loc_pose_{s}.json")) for s in ["validation", "test"]}
M = {s: json.load(open(f"artifacts/meta_labels_{s}.json")) for s in ["validation", "test"]}
cl = [c for c in E if c in L["validation"] and E[c].get("e") is not None]
e = np.array([E[c]["e"] for c in cl]); hum = np.array([E[c]["human"] for c in cl]); b = hum + e
v0 = np.array([E[c]["v0"] for c in cl]); nolead = np.array([E[c].get("lead_gap") is None for c in cl])
lp = L["validation"]; n = np.array([lp[c]["n"] for c in cl])
f = lambda k: np.array([np.nan if lp[c][k] is None else lp[c][k] for c in cl])
med, q25, q75, vm = f("d5_med"), f("d5_q25"), f("d5_q75"), f("v0_med")
grp = np.array([P["validation"][c]["log"] for c in cl])
print(f"n {len(cl)}, no-lead {nolead.sum()} ({(e[nolead]**2).sum()/(e**2).sum():.2f} of squared e), covered {(n>0).mean():.2f}")
print(f"corr(e, b - prior_med) all {np.corrcoef(e[n>0], (b-med)[n>0])[0,1]:.3f}, no-lead {np.corrcoef(e[(n>0)&nolead], (b-med)[(n>0)&nolead])[0,1]:.3f}")
# speed-scaled prior: neighbours' length rescaled by our speed / their speed is NOT used (mixes in v0 twice); plain features
X = np.c_[b, v0, np.log1p(n), med - b, q25 - b, q75 - b, vm - v0, nolead]
for name, mk in [("gbdt", lambda: HistGradientBoostingRegressor(loss="absolute_error", max_iter=200, learning_rate=0.05, max_depth=3)),
                 ("huber(med-b only)", None)]:
    pred = np.zeros(len(cl))
    for tr, te in GroupKFold(5).split(X, e, grp):
        if mk:
            m = mk().fit(X[tr], e[tr]); pred[te] = m.predict(X[te])
        else:
            z = np.nan_to_num(med - b); ok = n[tr] > 0
            h = HuberRegressor().fit(np.c_[z[tr][ok]], e[tr][ok]); pred[te] = np.where(n[te] > 0, h.predict(np.c_[z[te]]), 0)
    r = e - pred
    for nm, msk in [("all", np.ones(len(cl), bool)), ("no-lead", nolead), ("no-lead covered", nolead & (n > 0))]:
        d = np.abs(e[msk]) - np.abs(r[msk]); bs = [d[np.random.default_rng(i).integers(0, len(d), len(d))].mean() for i in range(2000)]
        print(f"{name:18s} {nm:16s} n {msk.sum():4d}  |e| {np.abs(e[msk]).mean():.3f} -> {np.abs(r[msk]).mean():.3f}  "
              f"gain {d.mean():+.3f} [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]  rmse {np.sqrt((e[msk]**2).mean()):.3f} -> {np.sqrt((r[msk]**2).mean()):.3f}")
# date shift, inputs only: own current speed minus neighbours' median current speed
for s in ["validation", "test"]:
    d = [M[s][c]["v_m0"] - L[s][c]["v0_med"] for c in L[s] if L[s][c]["n"] and c in M[s] and L[s][c]["v0_med"] is not None]
    yrs = sorted({P[s][c]["log"][:7] for c in P[s]})
    print(f"{s}: v0 - neighbour v0 median {np.median(d):+.2f} m/s, mean {np.mean(d):+.2f} (n {len(d)}); log months {yrs[0]}..{yrs[-1]}")
