"""Per-subcategory answer models fitted on labelled validation VQA.

Categorical subcategories: classify the ANSWER TEXT from question-text features,
then pick the choice whose text matches. Distance: regress the distance and pick
the numerically nearest choice.

Split is by CLIP so a model is never evaluated on clips it was fitted on.
Overall accuracy is reweighted to the TEST subcategory mix.
"""
import json, os, re, sys, collections, random, pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qfeat import load, norm_text, feats  # noqa

NUM = ["x", "y", "r", "az", "bx", "by", "bw", "bh", "barea",
       "c_v", "c_ax", "c_ay", "c_vh", "c_dv", "c_omega",
       "c_route_left", "c_turn", "c_curv", "g_ground", "g_size"]
CAT = ["cam", "range", "dir", "obj", "act_lon", "act_lat", "c_cmd"]
DIST_SUB = "distance_from_2d_bbox"


def vec(f, maps):
    v = [f.get(k, np.nan) for k in NUM]
    for k in CAT:
        m = maps[k]
        v.append(m.get(f.get(k), -1))
    return v


def build_maps(qs):
    maps = {}
    for k in CAT:
        vals = sorted({q["_f"].get(k) for q in qs if q["_f"].get(k) is not None})
        maps[k] = {v: i for i, v in enumerate(vals)}
    return maps


def num_of(s):
    m = re.search(r"(-?\d+(?:\.\d+)?)", str(s))
    return float(m.group(1)) if m else None


def usable_cols(X):
    """Columns with at least 2 distinct finite values.

    A column that is entirely NaN or constant within a subcategory (e.g. bbox
    features for questions that state only the 3D position) crashes the
    histogram binner, so the mask is fitted here and carried with the model.
    """
    keep = []
    for j in range(X.shape[1]):
        c = X[:, j]
        c = c[np.isfinite(c)]
        if c.size and np.unique(c).size >= 2:
            keep.append(j)
    return np.array(keep, int)


def fit_subcat(sub, rows, maps):
    X = np.array([vec(q["_f"], maps) for q in rows], float)
    cols = usable_cols(X)
    if cols.size == 0: return None
    X = X[:, cols]
    if sub == DIST_SUB:
        y = np.array([num_of(q.get("answer_text")) for q in rows], float)
        ok = ~np.isnan(y)
        if ok.sum() < 50: return None
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                          random_state=0)
        m.fit(X[ok], np.log1p(np.clip(y[ok], 0, None)))
        return ("reg", m, cols)
    y = np.array([norm_text(q.get("answer_text")) for q in rows])
    if len(set(y)) < 2 or len(rows) < 50: return None
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                       random_state=0)
    m.fit(X, y)
    return ("clf", m, cols)


def predict_letter(kind_model, q, maps, prior):
    ch = q.get("choices") or {}
    if not ch: return "A"
    kind, m, cols = kind_model
    x = np.array([vec(q["_f"], maps)], float)[:, cols]
    if kind == "reg":
        d = float(np.expm1(m.predict(x)[0]))
        cand = {k: num_of(v) for k, v in ch.items()}
        cand = {k: v for k, v in cand.items() if v is not None}
        if not cand: return "C" if "C" in ch else sorted(ch)[0]
        return min(cand, key=lambda k: abs(cand[k] - d))
    probs = m.predict_proba(x)[0]
    order = np.argsort(-probs)
    cls = m.classes_
    texts = {k: norm_text(v) for k, v in ch.items()}
    for i in order:                       # highest-probability text present as a choice
        hit = [k for k, t in texts.items() if t == cls[i]]
        if hit: return hit[0]
    return max(ch, key=lambda k: prior.get(texts[k], 0.0)) if prior else sorted(ch)[0]


def main():
    qs = load()
    # merge per-clip context (ego kinematics + route geometry) into the features
    cf = json.load(open("artifacts/clipfeat_validation.json"))
    from dist_geo import geo_feats          # calibration geometry (distance questions)
    for q in qs:
        q["_f"].update(cf.get(q["_clip"], {}))
        q["_f"].update(geo_feats(q, "cache/validation"))
    mix = collections.Counter()
    for d in sorted(os.listdir("cache/test")):
        p = os.path.join("cache/test", d, "reasoning_questions.json")
        if os.path.isfile(p):
            for q in (json.load(open(p)).get("questions") or []):
                mix[q.get("subcategory")] += 1
    tot = sum(mix.values()); mix = {k: v / tot for k, v in mix.items()}

    # Split by DRIVE LOG, not clip: validation has 1,000 clips from 506 logs and
    # sibling clips share scenes. A clip-level split leaked across siblings and
    # inflated local accuracy (0.81 local vs 0.597 on the server).
    log_of = lambda c: "_".join(c.split("_")[:2])
    logs = sorted({log_of(q["_clip"]) for q in qs})
    random.Random(0).shuffle(logs)
    half = set(logs[:len(logs)//2])
    fit = [q for q in qs if log_of(q["_clip"]) in half]
    ev  = [q for q in qs if log_of(q["_clip"]) not in half]
    maps = build_maps(qs)

    priors = collections.defaultdict(collections.Counter)
    for q in fit: priors[q["subcategory"]][norm_text(q.get("answer_text"))] += 1
    priors = {k: {t: c/sum(v.values()) for t, c in v.items()} for k, v in priors.items()}

    by_fit = collections.defaultdict(list)
    for q in fit: by_fit[q["subcategory"]].append(q)
    models = {}
    for sub, rows in by_fit.items():
        m = fit_subcat(sub, rows, maps)
        if m: models[sub] = m

    print(f"{'subcategory':34} {'n':>5} {'test%':>6} {'prior':>8} {'model':>8} {'best':>8}")
    tot_p = tot_m = tot_b = w_sum = 0.0
    chosen = {}
    for sub in sorted(set(mix) | {q['subcategory'] for q in ev}):
        rows = [q for q in ev if q["subcategory"] == sub]
        if not rows: continue
        w = mix.get(sub, 0.0); pri = priors.get(sub, {})
        a_p = np.mean([max(q["choices"], key=lambda k: pri.get(norm_text(q["choices"][k]), 0.0))
                       == q["answer"] for q in rows]) if pri else 0.25
        if sub in models:
            a_m = np.mean([predict_letter(models[sub], q, maps, pri) == q["answer"] for q in rows])
        else:
            a_m = float("nan")
        best = max(a_p, a_m if a_m == a_m else -1)
        chosen[sub] = "model" if (a_m == a_m and a_m > a_p) else "prior"
        tot_p += w*a_p; tot_m += w*(a_m if a_m == a_m else a_p); tot_b += w*best; w_sum += w
        print(f"{sub:34} {len(rows):>5} {100*w:>5.1f}% {a_p:>8.3f} "
              f"{'   n/a' if a_m != a_m else f'{a_m:>8.3f}'} {best:>8.3f}  <- {chosen[sub]}")
    print(f"\n{'TEST-WEIGHTED OVERALL':34} {'':>5} {100*w_sum:>5.1f}% {tot_p/w_sum:>8.3f} "
          f"{tot_m/w_sum:>8.3f} {tot_b/w_sum:>8.3f}")

    if os.environ.get("EVAL_ONLY"):
        return
    # refit on EVERYTHING for the shipped artifact
    by_all = collections.defaultdict(list)
    for q in qs: by_all[q["subcategory"]].append(q)
    final = {}
    for sub, rows in by_all.items():
        if chosen.get(sub) != "model": continue
        m = fit_subcat(sub, rows, maps)
        if m: final[sub] = m
    import time, hashlib
    os.makedirs("artifacts/versioned", exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    blob = {"models": final, "maps": maps, "use_model": chosen,
            "fitted_utc": stamp, "n_questions": len(qs), "source": "vqa_val"}
    vpath = f"artifacts/versioned/reason_models_{stamp}.pkl"
    with open(vpath, "wb") as fh:          # timestamped, never overwritten
        pickle.dump(blob, fh)
    with open("artifacts/reason_models.pkl", "wb") as fh:   # convenience pointer
        pickle.dump(blob, fh)
    print(f"   versioned copy -> {vpath}")
    print(f"\nshipped models for: {sorted(final)}")
    print("-> artifacts/reason_models.pkl")


if __name__ == "__main__":
    main()
