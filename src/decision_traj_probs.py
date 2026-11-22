"""Driving-decision probabilities from the PLANNER's predicted future (2026-11-13; planning thread's v9.9 Edge,
epoch-2 update-2400). Per-type HGB fitted on TRAIN questions with the TRUE future (src/decision_traj.py features),
applied to the planner's predicted future on validation (held-out: Edge trained on train only) and test.
-> artifacts/dec_traj_probs_{validation,test}.json {question_id: {letter: p}}
usage: python src/decision_traj_probs.py <val npz> <test npz>
"""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decision_traj import SUBS, clip_rows, questions
from sklearn.ensemble import HistGradientBoostingClassifier


def pred_map(path):
    z = np.load(path, allow_pickle=True); return {str(n): t for n, t in zip(z["names"], z["traj"])}


def test_questions():
    out = []
    for c in sorted(os.listdir("cache/test")):
        p = f"cache/test/{c}/reasoning_questions.json"
        if os.path.isfile(p):
            out += [(c, q) for q in json.load(open(p))["questions"] if q["subcategory"] in SUBS]
    return out


ck = "artifacts/dec_traj_train_feats.pkl"
if os.path.isfile(ck): TR = pickle.load(open(ck, "rb"))
else:
    TR = {}
    for c, q in questions("train"):
        if c not in TR: TR[c] = clip_rows("train", c)
    pickle.dump(TR, open(ck, "wb"))
tr = questions("train")
PV, PT = pred_map(sys.argv[1]), pred_map(sys.argv[2])
sets = {"validation": (questions("validation"), PV), "test": (test_questions(), PT)}
feat = {s: {} for s in sets}
for s, (qs, P) in sets.items():
    for c, _ in qs:
        if c not in feat[s] and c in P: feat[s][c] = clip_rows(s, c, P[c])
OUT = {s: {} for s in sets}
for sub in SUBS:
    X = [list(TR[c].values()) for c, q in tr if q["subcategory"] == sub and TR.get(c)]
    y = [q["choices"][q["answer"]] for c, q in tr if q["subcategory"] == sub and TR.get(c)]
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=0).fit(np.array(X), y)
    for s, (qs, P) in sets.items():
        n = ok = 0
        for c, q in qs:
            if q["subcategory"] != sub or not feat[s].get(c): continue
            p = dict(zip(m.classes_, m.predict_proba(np.array([list(feat[s][c].values())]))[0]))
            pr = {L: p.get(t, 0.0) + 1e-6 for L, t in q["choices"].items()}; z = sum(pr.values())
            OUT[s][q["question_id"]] = {L: v / z for L, v in pr.items()}
            if "answer" in q: n += 1; ok += max(pr, key=pr.get) == q["answer"]
        print(f"{sub:30s} {s:10s} n {sum(1 for c, q in qs if q['subcategory'] == sub and feat[s].get(c))}" + (f" acc {ok / n:.3f}" if n else ""), flush=True)
for s in sets: json.dump(OUT[s], open(f"artifacts/dec_traj_probs_{s}.json", "w"))
