"""No-vision reference on the 148-candidate vocabulary.

Same gradient-boosted per-sub-metric heads as scorer_v2, trained on TRAIN
labels148 shards, selection tuned on 15% held-out TRAIN, evaluated on all of
VALIDATION. This is the bar the vision scorer has to clear on the SAME
vocabulary: the vocabulary change alone (stopped clips can now pull away) may
capture headroom without any camera.
"""
import glob, os, sys, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def load(split, root="artifacts/labels148", kf=None):
    # key-frame shards only: "train_000of040.npz"; extra frames are "train_kf040_000of040.npz"
    pat = f"{root}/{split}_[0-9]*.npz" if kf is None else f"{root}/{split}_kf{kf:03d}_[0-9]*.npz"
    ps = [np.load(f) for f in sorted(glob.glob(pat))]
    assert ps, pat
    d = {k: np.concatenate([p[k] for p in ps]) for k in ("names", "traj", "feat", "labels")}
    d["vocab"] = ps[0]["vocab"]; return d


PERC = os.environ.get("PERC", "")       # ORACLE perception features (experiment only): "clip", "clr", "clip,clr"
_PF = {}


def perc(d, split):
    """Annotation-derived features aligned to d['names']; NaN where missing."""
    if split not in _PF:
        z = np.load(f"artifacts/{os.environ.get('PERC_FILE', 'percfeat')}_{split}.npz"); _PF[split] = ({n: i for i, n in enumerate(z["names"])}, z)
    idx, z = _PF[split]; n = len(d["names"]); K = d["traj"].shape[1]
    parts = []
    if "clip" in PERC:
        a = np.full((n, z["clip"].shape[1]), np.nan, np.float32)
        for i, nm in enumerate(d["names"]):
            if nm in idx: a[i] = z["clip"][idx[nm]]
        parts.append(np.repeat(a[:, None, :], K, 1))
    if "clr" in PERC:
        b = np.full((n, K, 2), np.nan, np.float32)
        for i, nm in enumerate(d["names"]):
            if nm in idx: b[i] = z["clr"][idx[nm]]
        parts.append(b)
    return parts


def rows(d, split=None):
    n, K = d["labels"].shape[:2]
    tr = d["traj"].astype(np.float32)
    end = tr[:, :, -1, :]                                       # endpoint (x, y, yaw) in ego frame
    mid = tr[:, :, 5, :]
    extra = perc(d, split) if (PERC and split) else []
    X = np.concatenate([np.repeat(d["feat"][:, None, :], K, 1),
                        np.repeat(d["vocab"][None], n, 0), end, mid] + extra, -1).reshape(n * K, -1)
    return X, d["labels"].reshape(n * K, 5), n, K


def nps(y): return y[..., 0] * y[..., 1] * (0.3 * y[..., 2] + 0.2 * y[..., 3] + 0.5 * y[..., 4])


tr, va = load("train"), load("validation")
Xt, Yt, nt, K = rows(tr, "train"); Xv, Yv, nv, _ = rows(va, "validation")
assert not (PERC and os.environ.get("EXTRA_KF")), "oracle features exist for key frames only"
perm = np.random.RandomState(0).permutation(nt)
fit_c, cal_c = perm[:int(.85 * nt)], perm[int(.85 * nt):]
m_fit = np.isin(np.repeat(np.arange(nt), K), fit_c); m_cal = ~m_fit
print(f"train {nt} clips ({len(fit_c)} fit / {len(cal_c)} calib), validation {nv}, K={K}")
# Optional extra TRAINING frames (EXTRA_KF="40,60,..."): only for fit clips, never for
# calibration clips (their other frames would leak the same scene into calibration).
EXTRA = [int(k) for k in os.environ.get("EXTRA_KF", "").split(",") if k]
TAG = os.environ.get("TAG", "")
if EXTRA:
    fit_names = set(tr["names"][fit_c])
    Xs, Ys = [Xt[m_fit]], [Yt[m_fit]]
    for kf in EXTRA:
        d = load("train", kf=kf); keep = np.isin(d["names"], list(fit_names))
        d = {k: (v[keep] if k != "vocab" else v) for k, v in d.items()}
        X_, Y_, n_, _ = rows(d); Xs.append(X_); Ys.append(Y_)
        print(f"  + frame {kf}: {n_} fit clips")
    Xf, Yf = np.concatenate(Xs), np.concatenate(Ys)
    Xt = np.concatenate([Xf, Xt[m_cal]]); Yt = np.concatenate([Yf, Yt[m_cal]])
    m_fit = np.r_[np.ones(len(Xf), bool), np.zeros(int(m_cal.sum()), bool)]; m_cal = ~m_fit
    print(f"  fit rows {len(Xf)//K} clip-frames")
t0 = time.time(); H = {}
for i, name in enumerate(["col", "da", "prog", "comf", "hum"]):
    y = Yt[m_fit, i]
    if name in ("col", "da", "comf"):
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=.06, max_leaf_nodes=63, random_state=0)
        m.fit(Xt[m_fit], np.round(y * 2).astype(int)); H[name] = m
    else:
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=.06, max_leaf_nodes=63, random_state=0)
        m.fit(Xt[m_fit], y); H[name] = m
print(f"heads fitted in {time.time()-t0:.0f}s")


def pred(X, n):
    o = {}
    for k, m in H.items():
        if k in ("col", "da", "comf"):
            p = m.predict_proba(X); o[k] = (p * (m.classes_[None] / 2.0)).sum(1).reshape(n, K)
        else:
            o[k] = np.clip(m.predict(X), 0, 1).reshape(n, K)
    return o


def S(p, w):
    e = 1e-6
    return w[0] * np.log(p["col"] + e) + w[1] * np.log(p["da"] + e) + np.log(.3 * p["prog"] + .2 * p["comf"] + .5 * p["hum"] + e)


pc = pred(Xt[m_cal], len(cal_c)); Nc = nps(Yt[m_cal].reshape(len(cal_c), K, 5))
best = max(((Nc[np.arange(len(Nc)), S(pc, (a, b)).argmax(1)].mean(), (a, b))
            for a in (0, .25, .5, 1, 2, 4, 8) for b in (0, .25, .5, 1, 2, 4, 8)))
pv = pred(Xv, nv); Nv = nps(Yv.reshape(nv, K, 5))
voc = va["vocab"]; cur = int(np.where((voc[:, 0] == 0) & (voc[:, 1] == 1.0) & (voc[:, 2] == 0))[0][0])
sel = S(pv, best[1]).argmax(1)
print(f"weights {best[1]} (calib NPS {best[0]:.4f})")
print(f"VALIDATION {nv} clips:  default {Nv[:, cur].mean():.4f}   no-vision selection {Nv[np.arange(nv), sel].mean():.4f}   "
      f"oracle {Nv.max(1).mean():.4f}")
st = va["feat"][:, 0] < 0.5
print(f"  stopped clips ({st.sum()}): default {Nv[st, cur].mean():.4f}  selected {Nv[np.arange(nv), sel][st].mean():.4f}  "
      f"oracle {Nv[st].max(1).mean():.4f}")


if PERC:
    for nm in ("col", "da", "prog", "hum"):
        print(f"  mean |pred-label| {nm}: {np.abs(pv[nm].ravel() - Yv[:, ['col','da','prog','comf','hum'].index(nm)]).mean():.4f}")
    sys.exit(0)                # oracle features do not exist for test

# ---------------------------------------------------------------- test selections
import json, os
ps = [np.load(f) for f in sorted(glob.glob("artifacts/labels148/test_*.npz"))]
te = {k: np.concatenate([p[k] for p in ps]) for k in ("names", "traj", "feat")}
te["vocab"] = ps[0]["vocab"]
assert np.array_equal(te["vocab"], va["vocab"]), "test vocabulary differs"
nte = len(te["names"])
tr_ = te["traj"].astype(np.float32)
Xte = np.concatenate([np.repeat(te["feat"][:, None, :], K, 1), np.repeat(te["vocab"][None], nte, 0),
                      tr_[:, :, -1, :], tr_[:, :, 5, :]], -1).reshape(nte * K, -1)
pte = pred(Xte, nte)
pick = S(pte, best[1]).argmax(1)
np.savez_compressed(f"artifacts/preds_gbdt148{TAG}.npz", eval_names=va["names"], test_names=te["names"],
                    **{f"eval_{k}": v for k, v in pv.items()}, **{f"test_{k}": v for k, v in pte.items()})
stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
out = f"artifacts/versioned/selections_test_novision148{TAG}_{stamp}.json"
json.dump({"split": "test", "cands": "labels148", "model": "gbdt_novision_148", "w": best[1],
           "val_default": float(Nv[:, cur].mean()), "val_selected": float(Nv[np.arange(nv), sel].mean()),
           "default_index": cur, "selections": {str(n): int(k) for n, k in zip(te["names"], pick)},
           "changed_from_default": int((pick != cur).sum())}, open(out, "w"))
st_te = te["feat"][:, 0] < 0.5
print(f"TEST {nte} route clips: {(pick != cur).sum()} changed from default; stopped {st_te.sum()} "
      f"(changed {(pick[st_te] != cur).sum()}) -> {out}")
