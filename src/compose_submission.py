"""Compose a submission from trajectories of one file and answers of another (the two halves are
scored independently). Keeps the trajectory file's clip order and trajectories byte-for-byte.
usage: python src/compose_submission.py <traj_from.json> <answers_from.json> <out.json> <method-name>"""
import json, sys, hashlib
traj_f, ans_f, out, name = sys.argv[1:5]
T, A = json.load(open(traj_f)), json.load(open(ans_f))
ans = {c["clip"]: c["answers"] for c in A["clips"]}
assert set(ans) == {c["clip"] for c in T["clips"]}, "clip sets differ"
for c in T["clips"]:
    old = {x["question_id"] for x in c["answers"]}; new = {x["question_id"] for x in ans[c["clip"]]}
    assert old == new, f"question ids differ for {c['clip']}"
    c["answers"] = ans[c["clip"]]
T["meta"] = dict(T.get("meta", {}), method=name, trajectories_from=traj_f.split("/")[-1], answers_from=ans_f.split("/")[-1])
json.dump(T, open(out, "w"))
print(out, hashlib.sha256(open(out, "rb").read()).hexdigest())
