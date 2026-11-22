"""Decision-theoretic arc-length prediction.

The reward is flat-topped (any FDE <= 1 m scores 1.0), saturates at 8 m, and is
asymmetric (progress is clamped above, so overshoot is cheaper than undershoot).
A point regressor optimising MSE or MAE is therefore mis-specified.

Instead: estimate the conditional distribution P(arc | x) with an ensemble of
quantile regressors, then choose
    pred* = argmax_p  E_arc[ 0.3*progress(p, arc) + 0.5*human(|p-arc|, lat) ]
which is the Bayes-optimal action under the actual metric.
"""
import json, os, pickle, sys, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arc_model import FEATS, X_of, arc_of, lat_of, human, reward

QS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


def fit_quantiles(X, y, qs=QS, **kw):
    ms = {}
    for q in qs:
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300,
                                          learning_rate=0.06, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0, **kw)
        m.fit(X, y)
        ms[q] = m
    return ms


def sample_matrix(ms, X):
    """(n_clips, n_quantiles) draws approximating P(arc | x)."""
    S = np.column_stack([ms[q].predict(X) for q in sorted(ms)])
    return np.maximum(np.sort(S, axis=1), 0.0)


def bayes_action(S, lat_hat, grid_extra=np.arange(-6.0, 6.01, 0.5)):
    """For each row pick p maximising mean reward over that row's quantile draws."""
    out = np.empty(len(S))
    for i in range(len(S)):
        s = S[i]
        cands = np.unique(np.concatenate([s, s.mean() + grid_extra, [0.0]]))
        cands = cands[cands >= 0.0]
        prog = np.where(s < 0.1, 1.0, np.clip(cands[:, None] / np.where(s < 0.1, 1.0, s)[None, :], 0, 1))
        fde = np.hypot(cands[:, None] - s[None, :], lat_hat)
        r = (0.3 * prog + 0.5 * human(fde)).mean(axis=1)
        out[i] = cands[int(np.argmax(r))]
    return out


def main():
    tr = [r for r in json.load(open("artifacts/arc_train.json")) if r["has_route"]]
    va_all = json.load(open("artifacts/arc_validation.json"))
    va = [r for r in va_all if r["has_route"]]
    Xtr, atr, ltr = X_of(tr), arc_of(tr), lat_of(tr)
    Xva, ava, lva = X_of(va), arc_of(va), lat_of(va)
    lat_hat = float(np.median(ltr))
    print(f"train {len(tr)} | validation {len(va)} | lat_hat={lat_hat:.2f} m")

    def rep(name, p):
        r = reward(p, ava, lva)
        h = human(np.hypot(ava - np.maximum(p, 0), lva))
        print(f"  {name:46} {r.mean():>8.4f} {h.mean():>8.4f} "
              f"{np.median(np.abs(ava - np.maximum(p, 0))):>9.2f}")
        return r.mean()

    print(f"\n  {'method':46} {'reward':>8} {'human':>8} {'|err|p50':>9}")
    base = np.maximum(Xva[:, 0]*5 + 0.5*0.75*Xva[:, 1]*25 - 1.25, 0)
    rep("kinematic (tuned 2-param)", base)

    ms = fit_quantiles(Xtr, atr)
    Sva = sample_matrix(ms, Xva)
    rep("quantile ensemble, median draw", Sva[:, len(QS)//2])
    pb = bayes_action(Sva, lat_hat)
    rep("BAYES action under the real reward", pb)

    # how much of the gap is the model vs the metric?
    Str = sample_matrix(ms, Xtr)
    print(f"\n  in-sample (train) bayes reward : "
          f"{reward(bayes_action(Str, lat_hat), atr, ltr).mean():.4f}")
    print(f"  ORACLE arc length (validation)  : {reward(ava, ava, lva).mean():.4f}")
    print(f"  => if in-sample ~= out-of-sample, ego-state features have hit an")
    print(f"     INFORMATION ceiling and the rest needs scene context (vision).")

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    os.makedirs("artifacts/versioned", exist_ok=True)
    blob = {"quantile_models": ms, "feats": FEATS, "lat_hat": lat_hat,
            "qs": QS, "n_train": len(tr), "fitted_utc": stamp}
    p = f"artifacts/versioned/arc_bayes_{stamp}.pkl"
    pickle.dump(blob, open(p, "wb")); pickle.dump(blob, open("artifacts/arc_bayes.pkl", "wb"))
    print(f"\n  -> {p}")


if __name__ == "__main__":
    main()
