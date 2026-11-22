"""Hydra-MDP-style trajectory scorer for nuReasoning NPS.

Per the literature, three design choices are not optional:
  1. Predict each SUB-METRIC separately. Distilling the aggregate score measured
     WORSE than distilling components (Hydra-MDP: 80.2 vs 83.0).
  2. Select with a LOG-LINEAR combination whose weights are grid-searched and
     deliberately differ from the metric's own weights, over-weighting the
     multiplicative gates. This is what bought Hydra-MDP +2.7 PDMS, of which
     +4.4 was drivable-area compliance alone.
  3. Select FROM the vocabulary. Continuous search against a vocabulary-trained
     scorer reward-hacks (TOAD: 34.7 -> 23.9 EPDMS).

Splits are by CLIP throughout.
"""
import argparse, json, os, pickle, sys, time, itertools
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arc_model import FEATS

SUBS = ["col", "da", "prog", "comf", "hum"]


def build_xy(data):
    vocab = data["vocab"]; rows = data["rows"]
    K = len(vocab)
    X, Y, clip_idx = [], [], []
    for ci, r in enumerate(rows):
        base = [r["f"].get(k, np.nan) for k in FEATS]
        for k, (scale, lat) in enumerate(vocab):
            X.append(base + [scale, lat, scale * r["L0"]])
            Y.append(r["y"][k])
            clip_idx.append(ci)
    return np.array(X, float), np.array(Y, float), np.array(clip_idx), K, vocab


def usable_cols(X):
    keep = []
    for j in range(X.shape[1]):
        c = X[:, j][np.isfinite(X[:, j])]
        if c.size and np.unique(c).size >= 2:
            keep.append(j)
    return np.array(keep, int)


def fit_heads(X, Y, cols):
    heads = {}
    Xc = X[:, cols]
    for i, name in enumerate(SUBS):
        y = Y[:, i]
        if name in ("col", "da"):                 # discrete -> classify, use E[score]
            m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08,
                                               max_leaf_nodes=31, random_state=0)
            m.fit(Xc, np.round(y * 2).astype(int))
            heads[name] = ("clf", m)
        else:
            m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08,
                                              max_leaf_nodes=31, random_state=0)
            m.fit(Xc, y)
            heads[name] = ("reg", m)
    return heads


def predict(heads, X, cols):
    Xc = X[:, cols]
    out = {}
    for name, (kind, m) in heads.items():
        if kind == "clf":
            p = m.predict_proba(Xc)
            out[name] = (p * (m.classes_[None, :] / 2.0)).sum(axis=1)   # expected score
        else:
            out[name] = np.clip(m.predict(Xc), 0.0, 1.0)
    return out


def select(pred, K, w):
    """Log-linear selection over the vocabulary; returns chosen index per clip."""
    eps = 1e-6
    w_nc, w_da, w_wt = w
    s = (w_nc * np.log(pred["col"] + eps)
         + w_da * np.log(pred["da"] + eps)
         + w_wt * np.log(0.3 * pred["prog"] + 0.2 * pred["comf"] + 0.5 * pred["hum"] + eps))
    return s.reshape(-1, K).argmax(axis=1)


def true_nps(Y, K):
    y = Y.reshape(-1, K, len(SUBS))
    return y[:, :, 0] * y[:, :, 1] * (0.3*y[:, :, 2] + 0.2*y[:, :, 3] + 0.5*y[:, :, 4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="artifacts/vocab_labels_val.json")
    a = ap.parse_args()
    data = json.load(open(a.labels))
    X, Y, ci, K, vocab = build_xy(data)
    n_clips = ci.max() + 1
    print(f"clips {n_clips}  vocabulary {K}  rows {len(X)}")

    rng = np.random.RandomState(0); perm = rng.permutation(n_clips)
    tr_c = set(perm[:int(n_clips*0.6)].tolist())
    ca_c = set(perm[int(n_clips*0.6):int(n_clips*0.8)].tolist())
    ev_c = set(perm[int(n_clips*0.8):].tolist())
    m_tr = np.isin(ci, list(tr_c)); m_ca = np.isin(ci, list(ca_c)); m_ev = np.isin(ci, list(ev_c))
    cols = usable_cols(X[m_tr])
    t0 = time.time()
    heads = fit_heads(X[m_tr], Y[m_tr], cols)
    print(f"fitted {len(SUBS)} heads in {time.time()-t0:.0f}s on {m_tr.sum()//K} clips")

    NPS_ca, NPS_ev = true_nps(Y[m_ca], K), true_nps(Y[m_ev], K)
    p_ca, p_ev = predict(heads, X[m_ca], cols), predict(heads, X[m_ev], cols)

    cur = int(np.where([v == (1.0, 0.0) for v in map(tuple, vocab)])[0][0])
    print(f"\n  current planner (vocab index {cur})   eval NPS = {NPS_ev[:, cur].mean():.4f}")
    print(f"  ORACLE best-in-vocabulary            eval NPS = {NPS_ev.max(axis=1).mean():.4f}")

    # grid-search log-linear weights on the CALIBRATION clips only
    # NOTE: the log-linear score is scale-invariant, so only RATIOS matter.
    # w_wt is pinned to 1.0; searching all three independently explores nothing.
    best = None
    for w_nc in (0, 0.25, 0.5, 1, 2, 4, 8, 16):
        for w_da in (0, 0.25, 0.5, 1, 2, 4, 8, 16):
            for w_wt in (1.0,):
                idx = select(p_ca, K, (w_nc, w_da, w_wt))
                v = NPS_ca[np.arange(len(idx)), idx].mean()
                if best is None or v > best[0]:
                    best = (v, (w_nc, w_da, w_wt))
    print(f"  best weights on calibration: nc={best[1][0]} da={best[1][1]} wt={best[1][2]}"
          f"  (calib NPS {best[0]:.4f})")

    for name, w in [("metric's own weights (1,1,1)", (1, 1, 1)), ("grid-searched", best[1])]:
        idx = select(p_ev, K, w)
        sel = NPS_ev[np.arange(len(idx)), idx].mean()
        print(f"  selection with {name:28} eval NPS = {sel:.4f}")

    idx = select(p_ev, K, best[1])
    sel = NPS_ev[np.arange(len(idx)), idx].mean()
    print(f"\n  gain over current planner: {sel - NPS_ev[:, cur].mean():+.4f}")
    print(f"  fraction of oracle headroom captured: "
          f"{100*(sel - NPS_ev[:, cur].mean())/max(NPS_ev.max(axis=1).mean()-NPS_ev[:, cur].mean(),1e-9):.1f}%")

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    os.makedirs("artifacts/versioned", exist_ok=True)
    blob = {"heads": heads, "cols": cols, "vocab": vocab, "weights": best[1],
            "feats": FEATS, "fitted_utc": stamp}
    p = f"artifacts/versioned/scorer_{stamp}.pkl"
    pickle.dump(blob, open(p, "wb")); pickle.dump(blob, open("artifacts/scorer.pkl", "wb"))
    print(f"  -> {p}")


if __name__ == "__main__":
    main()
