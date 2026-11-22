"""Reasoning models fitted on TRAIN-generated questions (devkit generator on 9,890
train clips) instead of only validation. Evaluated on the SAME held-out validation
logs as reason_model.py (seed-0 log split), test-weighted:
  (a) fit on validation fit-half  (current shipped recipe)
  (b) fit on train questions
  (c) fit on train + validation fit-half
Extra per-question features: calibration geometry (distance) and, when
artifacts/motion_<split>.json exists, DINOv3 correspondence motion (future motion).
"""
import collections, json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reason_model as RM
from qfeat import load
from dist_geo import geo_feats

MOTION = ["du", "dv", "s_best", "s_zero", "vx", "vy", "v", "v_radial", "k_best", "v_away_scale",
          "s_k0.8", "s_k0.9", "s_k1.12", "s_k1.25"]
RM.NUM = RM.NUM + [f"m_{k}" for k in MOTION]
COT = bool(os.environ.get("COTFEAT"))      # Alpamayo chain-of-causation cues as clip features
COT_SUBS = {"driving_decision_joint", "driving_decision_lateral", "driving_decision_longitudinal",
            "action_risk_assessment", "unsafe_action_identification"}     # scene-level questions only
if COT:
    from cotfeat import CUES
    RM.NUM = RM.NUM + [f"cot_{k}" for k in CUES]
IMG = bool(os.environ.get("IMGFEAT"))      # DINOv3 grid features per question (src/imgfeat_q.py)
if IMG:
    IMG_COLS = [str(c) for c in np.load("artifacts/imgfeat_train.npz")["cols"]]
    RM.NUM = RM.NUM + IMG_COLS


def _imgfeat(split):
    z = np.load(f"artifacts/imgfeat_{split}.npz")
    return {(str(c), str(q)): x for c, q, x in zip(z["clip"], z["qid"], z["X"].astype(np.float32))}
# sklearn turns early stopping on above 10k rows ("auto"); the validation-only models
# never had it, and its stratified split fails on 1-member answer classes. Pin it off.
import functools
# Plain boosting DIVERGES on the large train question sets (camera pair: training
# accuracy 0.71, held-out 0.29-0.78); L2 = 1.0 fixes it (held-out 0.989). Also drop
# answer classes with < MIN_CLASS examples.
L2 = float(os.environ.get("REASON_L2", "1.0")); MIN_CLASS = 10
RM.HistGradientBoostingClassifier = functools.partial(RM.HistGradientBoostingClassifier, early_stopping=False, l2_regularization=L2)
RM.HistGradientBoostingRegressor = functools.partial(RM.HistGradientBoostingRegressor, early_stopping=False, l2_regularization=L2)


def prep(root, split):
    qs = load(root)
    cf = json.load(open(f"artifacts/clipfeat_{split}.json"))
    cot = json.load(open(f"artifacts/cotfeat_{split}.json")) if COT else {}
    mp = f"artifacts/motion_{split}.json"
    mo = json.load(open(mp)) if os.path.isfile(mp) else {}
    im = _imgfeat(split) if IMG else {}
    for q in qs:
        v = im.get((q["_clip"], str(q.get("question_id"))))
        if v is not None: q["_f"].update({c: float(x) for c, x in zip(IMG_COLS, v) if x == x})
        q["_f"].update(cf.get(q["_clip"], {}))
        q["_f"].update(geo_feats(q, f"cache/{split}"))
        if COT and q.get("subcategory") in COT_SUBS: q["_f"].update(cot.get(q["_clip"], {}))
        m = mo.get(q.get("question_id"))
        if m and q.get("subcategory") == "future_motion_label":
            q["_f"].update({f"m_{k}": m.get(k, np.nan) for k in MOTION})
    print(f"{split}: {len(qs)} questions (motion features for {sum(1 for q in qs if 'm_du' in q['_f'])}, image features for {sum(1 for q in qs if 'img_obj0' in q['_f'] or 'img_scn0' in q['_f'])})", flush=True)
    return qs


def choice_probs(kind_model, q, maps):
    """P(choice letter) from a classifier subcategory model (None for regressors): class
    probabilities mapped onto the offered choice texts, renormalised over the offered choices."""
    kind, m, cols = kind_model
    ch = q.get("choices") or {}
    if kind == "reg" or not ch: return None
    pr = m.predict_proba(np.array([RM.vec(q["_f"], maps)], float)[:, cols])[0]
    p = {cl: float(v) for cl, v in zip(m.classes_, pr)}
    out = {k: p.get(RM.norm_text(v), 0.0) for k, v in ch.items()}
    z = sum(out.values())
    return {k: v / z for k, v in out.items()} if z > 0 else None


def fit_all(rows, maps):
    by = collections.defaultdict(list)
    for q in rows: by[q["subcategory"]].append(q)
    for sub, rr in by.items():
        if sub == RM.DIST_SUB: continue
        c = collections.Counter(RM.norm_text(q.get("answer_text")) for q in rr)
        by[sub] = [q for q in rr if c[RM.norm_text(q.get("answer_text"))] >= MIN_CLASS]
    models = {s: m for s, r in by.items() if (m := RM.fit_subcat(s, r, maps))}
    pri = collections.defaultdict(collections.Counter)
    for q in rows: pri[q["subcategory"]][RM.norm_text(q.get("answer_text"))] += 1
    pri = {k: {t: c / sum(v.values()) for t, c in v.items()} for k, v in pri.items()}
    return models, pri


def main():
    va = prep("vqa_val", "validation"); tr = prep("vqa_train", "train")
    mix = collections.Counter()
    for d in sorted(os.listdir("cache/test")):
        p = os.path.join("cache/test", d, "reasoning_questions.json")
        if os.path.isfile(p):
            for q in (json.load(open(p)).get("questions") or []): mix[q.get("subcategory")] += 1
    tot = sum(mix.values()); mix = {k: v / tot for k, v in mix.items()}
    log_of = lambda c: "_".join(c.split("_")[:2])
    logs = sorted({log_of(q["_clip"]) for q in va}); random.Random(0).shuffle(logs)
    half = set(logs[:len(logs) // 2])
    vfit = [q for q in va if log_of(q["_clip"]) in half]; ev = [q for q in va if log_of(q["_clip"]) not in half]
    trlogs = {log_of(q["_clip"]) for q in tr}
    print(f"held-out validation logs shared with train: {len({log_of(q['_clip']) for q in ev}) and len({log_of(q['_clip']) for q in ev} & trlogs)}")
    maps = RM.build_maps(va + tr)
    recipes = {"a_valhalf": vfit, "b_train": tr, "c_train+valhalf": tr + vfit}
    if os.environ.get("ONLY_C"): recipes = {"c_train+valhalf": tr + vfit}
    dump = os.environ.get("DUMP")      # per-question predictions of the last recipe on the held-out half
    R = {}
    for name, rows in recipes.items():  # (with ONLY_C the table below shows only recipe c)
        models, pri = fit_all(rows, maps); R[name] = {}
        for sub in sorted({q["subcategory"] for q in ev}):
            rr = [q for q in ev if q["subcategory"] == sub]; p = pri.get(sub, {})
            a_p = np.mean([max(q["choices"], key=lambda k: p.get(RM.norm_text(q["choices"][k]), 0.0)) == q["answer"] for q in rr]) if p else .25
            a_m = np.mean([RM.predict_letter(models[sub], q, maps, p) == q["answer"] for q in rr]) if sub in models else np.nan
            R[name][sub] = (len(rr), a_p, a_m)
        if dump:
            with open(dump, "w") as f:
                for q in ev:
                    sub = q["subcategory"]; p = pri.get(sub, {})
                    prior = max(q["choices"], key=lambda k: p.get(RM.norm_text(q["choices"][k]), 0.0)) if p else "A"
                    pm = RM.predict_letter(models[sub], q, maps, p) if sub in models else prior
                    cp = choice_probs(models[sub], q, maps) if sub in models else None
                    est = None              # regressor subcategories (distance): the raw estimate before snapping
                    if sub in models and models[sub][0] == "reg":
                        k_, m_, c_ = models[sub]; est = float(np.expm1(m_.predict(np.array([RM.vec(q["_f"], maps)], float)[:, c_])[0]))
                    f.write(json.dumps({"question_id": q.get("question_id"), "clip": q["_clip"], "subcategory": sub,
                                        "gold": q["answer"], "model": pm, "prior": prior, "probs": cp, "est": est, "recipe": name}) + "\n")
            print(f"dumped {len(ev)} held-out predictions -> {dump}", flush=True)
        print(f"fitted {name}", flush=True)
    print(f"\n{'subcategory':32} {'n':>5} {'test%':>6}" + "".join(f"{k:>18}" for k in recipes))
    tot = {k: 0.0 for k in recipes}; wsum = 0
    for sub in sorted(R[next(iter(R))]):
        w = mix.get(sub, 0.0); wsum += w
        cells = []
        for k in recipes:
            n, a_p, a_m = R[k][sub]; best = max(a_p, a_m if a_m == a_m else -1)
            tot[k] += w * best; cells.append(f"{best:>18.3f}")
        print(f"{sub:32} {R[next(iter(R))][sub][0]:>5} {100*w:>5.1f}%" + "".join(cells))
    print(f"{'TEST-WEIGHTED (best of prior/model)':45}" + "".join(f"{tot[k]/wsum:>18.3f}" for k in recipes))


if __name__ == "__main__" and not os.environ.get("SHIP"):
    main()


def ship():
    """Fit the shipped bundle on ALL train + validation questions (recipe c, validated
    above on held-out validation logs). Records NUM so the submission builder
    vectorises identically."""
    import pickle, time
    va = prep("vqa_val", "validation"); tr = prep("vqa_train", "train")
    rows = tr + va; maps = RM.build_maps(rows)
    models, pri = fit_all(rows, maps)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    blob = {"models": models, "maps": maps, "num": list(RM.NUM), "priors": pri, "l2": L2, "min_class": MIN_CLASS,
            "fitted_utc": stamp, "n_questions": len(rows), "source": "vqa_train + vqa_val (devkit generator)",
            "cot_subs": sorted(COT_SUBS) if COT else [], "img_cols": IMG_COLS if IMG else []}
    path = f"artifacts/versioned/reason_models_train+val{'_cot' if COT else ''}{'_img' if IMG else ''}_{stamp}.pkl"
    pickle.dump(blob, open(path, "wb")); print(f"{len(models)} subcategory models -> {path}")


if __name__ == "__main__" and os.environ.get("SHIP"):
    ship(); sys.exit(0)
