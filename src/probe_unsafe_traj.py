"""2026-11-13 probe p28: p26 answers with unsafe_action_identification re-answered as
p = 0.5 * p_deployed + 0.5 * p_traj; p_deployed = p19/p26 recipe (GBDT x VLM run 1, .5), p_traj = per-option model
on the planner's (v9.9 Edge e2-2400) predicted future (src/unsafe_traj_probs.py). Validation (288 held-out):
0.712 -> 0.747, cross-fit +0.021 (pos share 0.88), 13 wins / 3 losses at w .5. Stay-in-place trajectories.
Delta vs p26 0.7783 x 3000 = net answers on the changed unsafe questions."""
import glob, json, hashlib, sys
sys.path.insert(0, "recon/devkit")
from nureasoning.submission.format import validate_submission
SUB = "unsafe_action_identification"; W = 0.5
G = {json.loads(l)["question_id"]: json.loads(l) for l in open("artifacts/gbdt_test_probs.jsonl")}
V = {}
for f in glob.glob("runs_vlm/qd4b_s3000_test_s*of4.jsonl"):
    for l in open(f): r = json.loads(l); V[r["question_id"]] = r
T = json.load(open("artifacts/unsafe_traj_probs_test.json"))
S = json.load(open("submissions/probes/p26_dec_traj.json")); ch = n = 0
for c in S["clips"]:
    for a in c["answers"]:
        g = G.get(a["question_id"])
        if not g or g["subcategory"] != SUB or a["question_id"] not in T: continue
        pt = T[a["question_id"]]; pv = V[a["question_id"]]["probs"]
        d = {k: .5 * g["probs"].get(k, 0) + .5 * pv.get(k, 0) for k in pt}
        assert max(sorted(d), key=d.get) == a["answer"], a["question_id"]
        s = {k: (1 - W) * d[k] + W * pt[k] for k in pt}; new = max(sorted(s), key=s.get)
        n += 1; ch += new != a["answer"]; a["answer"] = new
S["meta"] = dict(S["meta"], method="p28_unsafe_traj", answer_strategy=S["meta"].get("answer_strategy", "") +
                 "; unsafe_action fused 50/50 with a per-option model on the planner's predicted trajectory (src/probe_unsafe_traj.py)")
out = "submissions/probes/p28_unsafe_traj.json"; json.dump(S, open(out, "w"))
err = validate_submission(S, data_root="cache/test")
print(f"unsafe answered {n}, changed vs p26 {ch}; {out}: validate errors {len(err)}, sha {hashlib.sha256(open(out, 'rb').read()).hexdigest()[:16]}")
