"""Chooser trained on the FINAL objective: rank the 148 candidates of a clip by NPS.

Baseline (v4.1): 5 per-sub-metric GBDT heads + hand-tuned log-linear combination.
Here: (a) one GBDT regressing NPS directly, (b) LightGBM LambdaRank with NPS grades
(NDCG@1 = quality of the single pick), (c) rank_xendcg. Same features, same train
fit/calib split (seed 0, 85/15), evaluated on the 849 validation route clips.
EXTRA_KF adds non-key training frames for fit clips (as in scorer148_novision).
"""
import glob, os, sys, time
import numpy as np
import lightgbm as lgb


def load(split, kf=None, root="artifacts/labels148"):
    pat = f"{root}/{split}_[0-9]*.npz" if kf is None else f"{root}/{split}_kf{kf:03d}_[0-9]*.npz"
    ps = [np.load(f) for f in sorted(glob.glob(pat))]
    d = {k: np.concatenate([p[k] for p in ps]) for k in ("names", "traj", "feat", "labels")}
    d["vocab"] = ps[0]["vocab"]; return d


def rows(d):
    n, K = d["labels"].shape[:2]; tr = d["traj"].astype(np.float32)
    X = np.concatenate([np.repeat(d["feat"][:, None, :], K, 1), np.repeat(d["vocab"][None], n, 0),
                        tr[:, :, -1, :], tr[:, :, 5, :]], -1)
    return X, d["labels"], n, K


nps = lambda y: y[..., 0] * y[..., 1] * (0.3 * y[..., 2] + 0.2 * y[..., 3] + 0.5 * y[..., 4])
tr, va = load("train"), load("validation")
Xt, Yt, nt, K = rows(tr); Xv, Yv, nv, _ = rows(va)
perm = np.random.RandomState(0).permutation(nt); fit_c, cal_c = perm[:int(.85 * nt)], perm[int(.85 * nt):]
Xf, Yf = [Xt[fit_c]], [Yt[fit_c]]
fit_names = set(tr["names"][fit_c])
for kf in [int(k) for k in os.environ.get("EXTRA_KF", "").split(",") if k]:
    d = load("train", kf); keep = np.isin(d["names"], list(fit_names))
    d = {k: (v[keep] if k != "vocab" else v) for k, v in d.items()}
    X_, Y_, _, _ = rows(d); Xf.append(X_); Yf.append(Y_)
Xf, Yf = np.concatenate(Xf), np.concatenate(Yf)
Xc, Yc = Xt[cal_c], Yt[cal_c]
Nf, Nc, Nv = nps(Yf), nps(Yc), nps(Yv)
flat = lambda X: X.reshape(-1, X.shape[-1])
print(f"fit clip-frames {len(Xf)}  calib {len(Xc)}  validation {nv}  K={K}", flush=True)

cur = 38
print(f"default {Nv[:, cur].mean():.4f}   oracle {Nv.max(1).mean():.4f}   [v4.1 GBDT heads: 0.5866]")
res = {}
base = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
            bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
for name, params, lab in [
    ("reg_nps", dict(objective="regression"), lambda N: N),
    ("lambdarank", dict(objective="lambdarank", eval_at=[1], lambdarank_truncation_level=20,
                        label_gain=list(range(31))), lambda N: np.round(N * 30).astype(int)),
    ("xendcg", dict(objective="rank_xendcg", eval_at=[1], label_gain=list(range(31))),
     lambda N: np.round(N * 30).astype(int)),
]:
    t0 = time.time()
    dtr = lgb.Dataset(flat(Xf), lab(Nf).ravel(), group=None if name == "reg_nps" else [K] * len(Xf))
    dca = lgb.Dataset(flat(Xc), lab(Nc).ravel(), group=None if name == "reg_nps" else [K] * len(Xc), reference=dtr)
    m = lgb.train({**base, **params}, dtr, num_boost_round=3000, valid_sets=[dca],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    def pick(X):
        return m.predict(flat(X), num_iteration=m.best_iteration).reshape(len(X), K).argmax(1)
    pc, pv = pick(Xc), pick(Xv)
    res[name] = (Nc[np.arange(len(Nc)), pc].mean(), Nv[np.arange(nv), pv].mean())
    st = va["feat"][:, 0] < 0.5
    print(f"  {name:10s} iters {m.best_iteration:4d}  calib {res[name][0]:.4f}  VALIDATION {res[name][1]:.4f}  "
          f"stopped {Nv[np.arange(nv), pv][st].mean():.4f}  ({time.time()-t0:.0f}s)", flush=True)
