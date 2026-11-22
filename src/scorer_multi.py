"""Chooser over the 148 route candidates + any number of path sources.

Each source is labelled by label_paths.py / label_a15.py into artifacts/labels_<dir>/.
Spec per source:  name|train=dirA|validation=dirB+dirC|test=dirB+dirC
(default: name|<split>=name for every split). On validation/test a source may come from
several models (e.g. both out-of-fold nuVLA halves): their candidates are all added.
On train each clip must have honest (out-of-fold) predictions for every source.
Features per candidate: route clip features, candidate vocab row, source one-hot,
that source's clip features (spread, own arc, kinematic arc), endpoint + midpoint.
Same per-sub-metric GBDT heads and log-linear selection as scorer148/175.

usage: SOURCES="a15raw;nuvla|train=nuvla_oof|validation=nuvla_h0+nuvla_h1|test=nuvla_h0+nuvla_h1" \
       python src/scorer_multi.py [--no-test] [--tag name]
"""
import argparse, glob, json, os, sys, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

nps = lambda y: y[..., 0] * y[..., 1] * (0.3 * y[..., 2] + 0.2 * y[..., 3] + 0.5 * y[..., 4])


def _cat(dirname, split):
    P = [np.load(f) for f in sorted(glob.glob(f"artifacts/{dirname}/{split}_[0-9]*.npz"))]
    assert P, f"no shards: artifacts/{dirname}/{split}_*"
    d = {k: np.concatenate([p[k] for p in P]) for k in P[0].files if k != "vocab"}
    d["vocab"] = P[0]["vocab"]; return d


def parse(spec):
    parts = spec.split("|"); name = parts[0]; m = {}
    for p in parts[1:]:
        k, v = p.split("="); m[k] = v.split("+")
    return name, m


def dirs_of(spec, split):
    name, m = parse(spec)
    return [] if m.get(split) == ["none"] else [f"labels_{x}" for x in m.get(split, [name])]


def load(split, sources):
    base = _cat("labels148", split)
    names = list(base["names"]); keep = set(names)
    blocks = []                                           # per source: list of dicts (one per model dir)
    for si, spec in enumerate(sources):
        name, m = parse(spec)
        # "train=none": the source has no predictions on this split (e.g. JEPA planners without train
        # frames). It then adds no dishes there and its clip features are NaN; with VALCV its dishes
        # are learned from the validation folds only.
        dirs = [] if m.get(split) == ["none"] else [f"labels_{d}" for d in m.get(split, [name])]
        ds = [_cat(d, split) for d in dirs]
        for d in ds: keep &= set(d["names"])
        blocks.append((si, ds))
    idx = [i for i, n in enumerate(names) if n in keep]
    nm = np.array(names)[idx]
    T = [base["traj"][idx]]; V = [np.concatenate([base["vocab"], np.zeros((148, 1 + len(sources) + 3), np.float32)], 1)]
    F = [np.zeros((len(idx), 148, 3), np.float32)]
    L = [base["labels"][idx]] if "labels" in base else None
    SRC = [np.full(148, -1)]; BLK = [["route148", "labels148", 148]]
    for si, ds in blocks:
        for d, dn in zip(ds, dirs_of(sources[si], split)):
            pos = {n: i for i, n in enumerate(d["names"])}; j = [pos[n] for n in nm]
            K = d["traj"].shape[1]
            onehot = np.zeros((K, len(sources)), np.float32); onehot[:, si] = 1
            V.append(np.concatenate([d["vocab"], np.ones((K, 1), np.float32), onehot, np.zeros((K, 3), np.float32)], 1))
            T.append(d["traj"][j]); F.append(np.repeat(d["a15feat"][j][:, None, :], K, 1))
            SRC.append(np.full(K, si)); BLK.append([parse(sources[si])[0], dn, int(K)])
            if L is not None: L.append(d["labels"][j])
    # every candidate also sees every source's clip features (mean over that source's models):
    # e.g. Alpamayo's own arc length is evidence for scoring the ROUTE candidates as well
    CF = []
    for si, ds in blocks:
        per = []
        for d in ds:
            pos = {n: i for i, n in enumerate(d["names"])}; per.append(d["a15feat"][[pos[n] for n in nm]])
        CF.append(np.mean(per, 0) if per else np.full((len(nm), 3), np.nan, np.float32))
    clipfeat = np.concatenate([base["feat"][idx]] + CF, 1) if CF else base["feat"][idx]
    if os.environ.get("COTFEAT"):              # Alpamayo chain-of-causation cues (src/cotfeat.py)
        cf = json.load(open(f"artifacts/cotfeat_{split}.json")); keys = sorted({k for v in cf.values() for k in v})
        clipfeat = np.concatenate([clipfeat, np.array([[cf.get(n, {}).get(k, np.nan) for k in keys] for n in nm], np.float32)], 1)
    out = {"names": nm, "feat": clipfeat, "traj": np.concatenate(T, 1),
           "vocab": np.concatenate(V), "srcfeat": np.concatenate(F, 1),
           "src": np.concatenate(SRC), "blocks": BLK}
    if L is not None: out["labels"] = np.concatenate(L, 1)
    return out


def rows(d):
    n, K = d["traj"].shape[:2]; tr = d["traj"].astype(np.float32)
    return np.concatenate([np.repeat(d["feat"][:, None, :], K, 1), d["srcfeat"], np.repeat(d["vocab"][None], n, 0),
                           tr[:, :, -1, :], tr[:, :, 5, :]], -1).reshape(n * K, -1), n, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-test", action="store_true"); ap.add_argument("--tag", default="multi")  # env SEED, VALCV, VALW, DOMAIN: see main()
    a = ap.parse_args()
    sources = [s for s in os.environ.get("SOURCES", "").split(";") if s]
    tr, va = load("train", sources), load("validation", sources)
    Xt, nt, Kt = rows(tr); Xv, nv, Kv = rows(va)
    Yt = tr["labels"].reshape(nt * Kt, 5)
    seed = int(os.environ.get("SEED", "0"))             # SEED varies heads, fit/cal split and folds
    perm = np.random.RandomState(seed).permutation(nt); fit_c, cal_c = perm[:int(.85 * nt)], perm[int(.85 * nt):]
    m_fit = np.isin(np.repeat(np.arange(nt), Kt), fit_c); m_cal = ~m_fit
    print(f"sources {sources}\ntrain {nt} clips K={Kt} | validation {nv} clips K={Kv}", flush=True)
    # VALCV=k: validation clips also become training data, cross-fitted in k folds grouped by
    # drive log, so every validation score still comes from heads that never saw that clip.
    # (Train OOF predictions come from the same period as the other half; validation is a new
    # period, so trust learned on train alone can be miscalibrated.) VALW weights val rows.
    # DOMAIN=1 adds a 0/1 train-vs-held-out-period column (1 for validation and test rows).
    kcv = int(os.environ.get("VALCV", "0")); vw = float(os.environ.get("VALW", "1"))
    dom = os.environ.get("DOMAIN") == "1"
    if dom:
        Xt = np.concatenate([Xt, np.zeros((len(Xt), 1), np.float32)], 1)
        Xv = np.concatenate([Xv, np.ones((len(Xv), 1), np.float32)], 1)
    Yv = va["labels"].reshape(nv * Kv, 5)

    # ENS=n: average n members (member e: GBDT random_state seed*1000+e, 90% of the train fit
    # clips); ENS=1 is exactly the single model. Heads are a list of members everywhere.
    ens = int(os.environ.get("ENS", "1"))

    def fit(Xextra=None, Yextra=None):
        Hs = []
        for e in range(ens):
            rs = seed if ens == 1 else seed * 1000 + e
            rows_ = m_fit if ens == 1 else (m_fit & np.isin(np.repeat(np.arange(nt), Kt),
                                            np.random.RandomState(rs).choice(fit_c, int(.9 * len(fit_c)), replace=False)))
            X, Y = Xt[rows_], Yt[rows_]; w = None             # None (not ones): identical to the old path
            if Xextra is not None:
                X = np.concatenate([X, Xextra]); Y = np.concatenate([Y, Yextra])
                w = None if vw == 1 else np.concatenate([np.ones(int(rows_.sum())), np.full(len(Xextra), vw)])
            H = {}
            for i, name in enumerate(["col", "da", "prog", "comf", "hum"]):
                if name in ("col", "da", "comf"):
                    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=.06, max_leaf_nodes=63, random_state=rs)
                    m.fit(X, np.round(Y[:, i] * 2).astype(int), sample_weight=w)
                else:
                    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=.06, max_leaf_nodes=63, random_state=rs)
                    m.fit(X, Y[:, i], sample_weight=w)
                H[name] = m
            Hs.append(H)
        return Hs
    t0 = time.time(); H = fit() if not kcv else None      # with VALCV the heads are fitted per fold
    print(f"heads fitted in {time.time()-t0:.0f}s (members {ens})", flush=True)

    def pred(X, n, K, H=None):
        Hs = H if H is not None else globals()["_H"]
        o = {}
        for Hm in Hs:
            for k, m in Hm.items():
                if k in ("col", "da", "comf"):
                    p = m.predict_proba(X); v = (p * (m.classes_[None] / 2.0)).sum(1).reshape(n, K)
                else:
                    v = np.clip(m.predict(X), 0, 1).reshape(n, K)
                o[k] = o.get(k, 0) + v / len(Hs)
        return o
    globals()["_H"] = H
    S = lambda p, w: w[0] * np.log(p["col"] + 1e-6) + w[1] * np.log(p["da"] + 1e-6) + \
        np.log(.3 * p["prog"] + .2 * p["comf"] + .5 * p["hum"] + 1e-6)
    Xc = Xt[m_cal]; Nc = nps(Yt[m_cal].reshape(len(cal_c), Kt, 5))
    def calib(H):
        pc = pred(Xc, len(cal_c), Kt, H)
        return max(((Nc[np.arange(len(Nc)), S(pc, (a_, b_)).argmax(1)].mean(), (a_, b_))
                    for a_ in (0, .25, .5, 1, 2, 4, 8) for b_ in (0, .25, .5, 1, 2, 4, 8)))
    best = calib(H) if H is not None else None; Nv = nps(va["labels"])
    if kcv:
        logs = [json.load(open(f"cache/validation/{n}/metadata.json")).get("log_name", n) for n in va["names"]]
        ul = sorted(set(logs)); fold_of_log = {l: i % kcv for i, l in enumerate(np.random.RandomState(1 + seed).permutation(ul))}
        fold = np.array([fold_of_log[l] for l in logs]); sel = np.zeros(nv, int)
        rowfold = np.repeat(fold, Kv)
        for f in range(kcv):
            tr_rows = rowfold != f
            Hf = fit(Xv[tr_rows], Yv[tr_rows]); bf = calib(Hf)
            te = np.where(fold == f)[0]
            sel[te] = S(pred(Xv[rowfold == f], len(te), Kv, Hf), bf[1]).argmax(1)
            print(f"  fold {f}: {len(te)} clips, weights {bf[1]}, held-out {Nv[te, sel[te]].mean():.4f}", flush=True)
        H = fit(Xv, Yv); best = calib(H); globals()["_H"] = H      # final heads for test: train + all validation
    else:
        pv = pred(Xv, nv, Kv); sel = S(pv, best[1]).argmax(1)
    share = {s.split("|")[0]: round(float(np.mean(va["src"][sel] == si)), 3) for si, s in enumerate(sources)}
    print(f"weights {best[1]} (calib {best[0]:.4f})")
    print(f"VALIDATION {nv} clips: default {Nv[:, 38].mean():.4f}  selected {Nv[np.arange(nv), sel].mean():.4f}  "
          f"oracle {Nv.max(1).mean():.4f}  picks by source {share}", flush=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    os.makedirs("artifacts/versioned", exist_ok=True)
    # validation picks (out-of-fold when VALCV) for error analysis; never used to fit anything
    json.dump({"split": "validation", "sources": sources, "tag": a.tag, "seed": seed, "ens": ens, "valcv": kcv,
               "val_selected": float(Nv[np.arange(nv), sel].mean()), "blocks": va["blocks"],
               "selections": {str(n): int(k) for n, k in zip(va["names"], sel)}},
              open(f"artifacts/versioned/valsel_{a.tag}_{stamp}.json", "w"))
    if a.no_test: return
    te = load("test", sources); Xe, ne, Ke = rows(te)
    if dom: Xe = np.concatenate([Xe, np.ones((len(Xe), 1), np.float32)], 1)
    pick = S(pred(Xe, ne, Ke), best[1]).argmax(1)
    out = f"artifacts/versioned/selections_test_{a.tag}_{stamp}.json"
    json.dump({"split": "test", "cands": "labels148+sources", "sources": sources, "model": f"gbdt_{a.tag}",
               "w": best[1], "seed": seed, "ens": ens, "valcv": kcv, "valw": vw, "domain": dom, "val_selected": float(Nv[np.arange(nv), sel].mean()),
               "selections": {str(n): int(k) for n, k in zip(te["names"], pick)},
               "candidate_count_test": int(Ke), "blocks": te["blocks"]}, open(out, "w"))
    print(f"TEST {ne} clips K={Ke} -> {out}")


if __name__ == "__main__":
    main()
