"""Fit the 5-second arc-length predictor against the REAL planning objective.

reward = 0.3*progress + 0.2*comfort + 0.5*human   (gates excluded: invisible on test)
  progress = clip(pred/gt, 0, 1), and 1.0 when gt < 0.1 m
  human    = smoothstep on FDE = hypot(pred - gt, lateral_offset_of_gt_from_route)

MSE is mis-specified here: the human term is flat-topped (anything within 1 m
scores 1.0) and saturates at 8 m, so squared error over-weights outliers that
cost nothing extra. We therefore fit several losses and pick by REWARD, then
apply a post-hoc affine calibration chosen on train, never on validation.
"""
import json, os, sys, pickle, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

FEATS = ["v0","ax","ay","vh_all","vh_last5","vh_last10","vh_min","vh_max",
         "dv_recent","dv_all","frac_slow","omega","cmd","route_left","has_route",
         "turn10","turn20","turn30","turn45","turn60","turn80",
         "curv10","curv20","curv30","curv45","curv60","curv80"]


def X_of(rows): return np.array([[r.get(k, np.nan) for k in FEATS] for r in rows], float)
def arc_of(rows): return np.array([r["arc"] for r in rows], float)
def lat_of(rows): return np.array([r["lat"] for r in rows], float)


def human(fde):
    f, z = 1.0, 8.0
    t = np.clip((fde - f) / (z - f), 0, 1)
    return np.where(fde <= f, 1.0, np.where(fde >= z, 0.0, 1 - t*t*(3 - 2*t)))


def reward(pred, arc, lat):
    pred = np.maximum(pred, 0.0)
    prog = np.where(arc < 0.1, 1.0, np.clip(pred / np.where(arc < 0.1, 1.0, arc), 0, 1))
    return 0.3*prog + 0.2 + 0.5*human(np.hypot(arc - pred, lat))


def calibrate(pred, arc, lat):
    """Affine (scale, bias) chosen to maximise reward. Fitted on TRAIN only."""
    best = (reward(pred, arc, lat).mean(), 1.0, 0.0)
    for s in np.arange(0.80, 1.21, 0.02):
        for b in np.arange(-4.0, 3.01, 0.25):
            r = reward(pred*s + b, arc, lat).mean()
            if r > best[0]: best = (r, s, b)
    return best[1], best[2]


def main():
    tr = json.load(open("artifacts/arc_train.json"))
    va = json.load(open("artifacts/arc_validation.json"))
    tr = [r for r in tr if r["has_route"]]
    va_all, va = va, [r for r in va if r["has_route"]]
    print(f"train {len(tr)} route clips | validation {len(va)} route clips "
          f"({len(va_all)} total)")

    Xtr, atr, ltr = X_of(tr), arc_of(tr), lat_of(tr)
    Xva, ava, lva = X_of(va), arc_of(va), lat_of(va)

    # kinematic baseline, previously grid-searched on validation
    base = np.maximum(Xva[:, 0]*5 + 0.5*0.75*Xva[:, 1]*25 - 1.25, 0)
    print(f"\n  {'model':44} {'reward':>8} {'human':>8} {'|err| p50':>10}")
    def rep(name, p):
        r = reward(p, ava, lva)
        h = human(np.hypot(ava - np.maximum(p,0), lva))
        print(f"  {name:44} {r.mean():>8.4f} {h.mean():>8.4f} "
              f"{np.median(np.abs(ava-np.maximum(p,0))):>10.2f}")
        return r.mean()
    rep("kinematic v0*5+0.375*ax*25-1.25 (tuned)", base)

    # inner split of TRAIN for calibration, so validation is never used to tune
    n = len(tr); idx = np.random.RandomState(0).permutation(n)
    cut = int(n*0.75); fit_i, cal_i = idx[:cut], idx[cut:]

    results = {}
    for loss, kw in [("squared_error", {}), ("absolute_error", {}),
                     ("quantile", {"quantile": 0.40}), ("quantile", {"quantile": 0.45}),
                     ("quantile", {"quantile": 0.50}), ("quantile", {"quantile": 0.55})]:
        tag = loss + (f"@{kw['quantile']}" if kw else "")
        m = HistGradientBoostingRegressor(loss=loss, max_iter=400, learning_rate=0.06,
                                          max_leaf_nodes=31, l2_regularization=1.0,
                                          random_state=0, **kw)
        m.fit(Xtr[fit_i], atr[fit_i])
        s, b = calibrate(m.predict(Xtr[cal_i]), atr[cal_i], ltr[cal_i])
        p = m.predict(Xva)*s + b
        r = rep(f"{tag}  (calib s={s:.2f} b={b:+.2f})", p)
        results[tag] = (r, m, s, b)

    best = max(results, key=lambda k: results[k][0])
    r, m, s, b = results[best]
    print(f"\n  BEST: {best}   reward {r:.4f}  vs kinematic {reward(base,ava,lva).mean():.4f}")
    print(f"  ORACLE arc length: {reward(ava, ava, lva).mean():.4f}")

    # refit best config on ALL train rows
    loss = best.split("@")[0]; kw = {"quantile": float(best.split("@")[1])} if "@" in best else {}
    final = HistGradientBoostingRegressor(loss=loss, max_iter=400, learning_rate=0.06,
                                          max_leaf_nodes=31, l2_regularization=1.0,
                                          random_state=0, **kw)
    final.fit(Xtr, atr)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    os.makedirs("artifacts/versioned", exist_ok=True)
    blob = {"model": final, "feats": FEATS, "scale": s, "bias": b,
            "loss": best, "n_train": len(tr), "fitted_utc": stamp}
    p = f"artifacts/versioned/arc_model_{stamp}.pkl"
    pickle.dump(blob, open(p, "wb")); pickle.dump(blob, open("artifacts/arc_model.pkl", "wb"))
    print(f"  -> {p}")


if __name__ == "__main__":
    main()
