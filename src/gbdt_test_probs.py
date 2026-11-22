"""Per-question choice probabilities of the shipped GBDT bundle (v7.1 recipe) on the TEST questions,
built exactly as make_submission.answer_for(q, "qmodel") builds features.
-> artifacts/gbdt_test_probs.jsonl {question_id, clip, subcategory, probs, pred}
usage: REASON_MODELS=<bundle.pkl> python src/gbdt_test_probs.py
"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("REASON_MODELS", "artifacts/versioned/reason_models_train+val_cot_20260927T040508Z.pkl")
import make_submission as M
from dist_geo import geo_feats

M._CLIPS_ROOT = "cache/test"
bundle = M._load_models(os.environ["REASON_MODELS"])
n = 0
with open("artifacts/gbdt_test_probs.jsonl", "w") as out:
    for cp in sorted(glob.glob("cache/test/*")):
        if not os.path.isfile(cp + "/reasoning_questions.json"): continue
        clip = os.path.basename(cp)
        for q in M.load_clip_questions(cp):
            q["_clip"] = clip; sub = q.get("subcategory"); mdl = bundle["models"].get(sub)
            if mdl is None or not q.get("choices"): continue
            qq = dict(q); f = M._qfeats(q); f.update(M._load_clipfeat().get(clip, {})); qq["_f"] = f
            f.update(geo_feats(qq, "cache/test")); f.update(M._motion_feats(q))
            if sub in set(bundle.get("cot_subs") or []): f.update(M._cot_feats(q))
            try: p = M._choice_probs(mdl, qq, bundle["maps"])
            except Exception: p = None
            if mdl[0] == "reg":             # distance: raw regressor estimate + the letter the bundle answers
                import numpy as np
                est = float(np.expm1(mdl[1].predict(np.array([M.RM.vec(qq["_f"], bundle["maps"])], float)[:, mdl[2]])[0]))
                out.write(json.dumps({"question_id": q["question_id"], "clip": clip, "subcategory": sub, "est": est,
                                      "pred": M.RM.predict_letter(mdl, qq, bundle["maps"], {})}) + "\n"); n += 1; continue
            if not p: continue
            out.write(json.dumps({"question_id": q["question_id"], "clip": clip, "subcategory": sub, "probs": p,
                                  "pred": max(p, key=p.get)}) + "\n"); n += 1
print(n, "questions -> artifacts/gbdt_test_probs.jsonl")
