"""2026-11-17 planning thread's sampled Edge decisions (K samples per clip, counts over exact MCQ option strings,
jepa-nureasoning tools/edge_dec_export.py) -> per-question letter probabilities for the three driving-decision types.
p(letter) = (count(option text) + A) / (K + A * n_options), A = 0.5. Key-frame questions only (vqa_val / cache test).
usage: python src/edge_dec_probs.py <export.jsonl> <split validation|test> <out.json>"""
import json, os, sys
from decision_traj import questions


def test_questions():        # as src/decision_traj_probs.py (not imported: that module runs on import)
    out = []
    for c in sorted(os.listdir("cache/test")):
        p = f"cache/test/{c}/reasoning_questions.json"
        if os.path.isfile(p): out += [(c, q) for q in json.load(open(p))["questions"]]
    return out


A = 0.5; KEY = {"driving_decision_longitudinal": "long", "driving_decision_lateral": "lat", "driving_decision_joint": "joint"}
E = {r["clip"]: r for r in map(json.loads, open(sys.argv[1]))}
qs = questions("validation") if sys.argv[2] == "validation" else test_questions()
out, miss, n = {}, 0, 0
for c, q in qs:
    if q["subcategory"] not in KEY or c not in E: continue
    cnt = E[c][KEY[q["subcategory"]]]; K = sum(cnt.values()) or 1
    hit = sum(cnt.get(t, 0) for t in q["choices"].values()); miss += K - hit; n += K
    pr = {L: (cnt.get(t, 0) + A) / (K + A * len(q["choices"])) for L, t in q["choices"].items()}
    z = sum(pr.values()); out[q["question_id"]] = {L: v / z for L, v in pr.items()}
json.dump(out, open(sys.argv[3], "w"))
print(f"{len(out)} questions from {len(E)} clips; samples not matching any option {miss} / {n}")
