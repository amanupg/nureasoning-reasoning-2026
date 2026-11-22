"""2026-11-15 bundle: answers from one submission + trajectories from another (separate board columns, so one file can
carry a reasoning probe and a planning candidate). Optional --ar_from: copy that file's action_risk answers where they
differ from --ar_base (re-applies p27's 56 action_risk flips onto newer answers). Devkit validate + round-trip checks.
usage: python src/bundle_sub.py --answers A.json --traj T.json --out O.json --method NAME [--ar_from p27 --ar_base p19]"""
import argparse, json, hashlib, os, sys
sys.path.insert(0, "recon/devkit")
from nureasoning.submission.format import validate_submission
ap = argparse.ArgumentParser()
for k in ("--answers", "--traj", "--out", "--method"): ap.add_argument(k, required=True)
ap.add_argument("--ar_from"); ap.add_argument("--ar_base")
a = ap.parse_args()
AR = {q["question_id"] for c in os.listdir("cache/test") if os.path.isfile(f"cache/test/{c}/reasoning_questions.json")
      for q in json.load(open(f"cache/test/{c}/reasoning_questions.json"))["questions"]
      if q["subcategory"] == "action_risk_assessment"}
ans = lambda S: {q["question_id"]: q["answer"] for c in S["clips"] for q in c["answers"]}
S, T = json.load(open(a.answers)), json.load(open(a.traj))
tr = {c["clip"]: c["trajectory"] for c in T["clips"]}
assert len(tr) == len(S["clips"]) == 1000 and T["meta"]["trajectory_steps"] == S["meta"]["trajectory_steps"]
flip = {}
if a.ar_from:
    F, B = ans(json.load(open(a.ar_from))), ans(json.load(open(a.ar_base))); cur = ans(S)
    assert all(cur[q] == B[q] for q in AR), "answers file's action_risk differs from --ar_base"
    flip = {q: F[q] for q in AR if F[q] != B[q]}
for c in S["clips"]:
    c["trajectory"] = tr[c["clip"]]
    for q in c["answers"]: q["answer"] = flip.get(q["question_id"], q["answer"])
S["meta"] = dict(S["meta"], method=a.method, planner=T["meta"].get("planner", os.path.basename(a.traj)),
                 answer_strategy=f"answers from {os.path.basename(a.answers)}" + (f" + {len(flip)} action_risk from "
                 f"{os.path.basename(a.ar_from)}" if flip else "") + f"; trajectories from {os.path.basename(a.traj)}")
json.dump(S, open(a.out, "w"))
R = json.load(open(a.out)); r, s0 = ans(R), ans(json.load(open(a.answers)))
assert all(c["trajectory"] == tr[c["clip"]] for c in R["clips"])
print(f"{a.out}: answers changed vs {os.path.basename(a.answers)} {sum(r[q] != s0[q] for q in r)} (action_risk flips "
      f"{len(flip)}), validate errors {len(validate_submission(R, data_root='cache/test'))}, "
      f"sha {hashlib.sha256(open(a.out, 'rb').read()).hexdigest()[:16]}")
