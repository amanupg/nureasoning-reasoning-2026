"""2026-11-15 probe p33: p31 answers with VLM run 1's probabilities replaced by the mean of run 1 and the camera run
(qd4b_r1fe) inside the deployed recipes, same weights: action_risk (GBDT x VLM .5), joint (.4) / lateral (.5) then
50/50 with planner-trajectory probs, unsafe (.5) then 50/50 planner. Longitudinal (a 0) and all other types unchanged.
Validation (src/p33_avg_eval.py, held-out, n 151/155/148/165): net +1 / 0 / -1 / -2, i.e. flat. Tests on the board
whether the camera run's raw gains survive fusion on test (validation has not tracked test for reasoning).
Stay-in-place trajectories. Delta vs p31 x 3000 = net answers on the changed questions.
usage: python src/probe_vlm_avg.py"""
import glob, json, hashlib, sys
from collections import Counter
sys.path.insert(0, "recon/devkit")
from nureasoning.submission.format import validate_submission
REC = {"action_risk_assessment": (.5, 0, None), "driving_decision_joint": (.4, .5, "dec"),
       "driving_decision_lateral": (.5, .5, "dec"), "unsafe_action_identification": (.5, .5, "unsafe")}
G = {json.loads(l)["question_id"]: json.loads(l) for l in open("artifacts/gbdt_test_probs.jsonl")}
def load(pat):
    D = {}
    for f in glob.glob(pat):
        for l in open(f): r = json.loads(l); D[r["question_id"]] = r
    return D
V1, V2 = load("runs_vlm/qd4b_s3000_test_s*of4.jsonl"), load("runs_vlm/qd4b_r1fe/test_s*of2.jsonl")
TR = {"dec": json.load(open("artifacts/dec_traj_probs_test.json")), "unsafe": json.load(open("artifacts/unsafe_traj_probs_test.json"))}
S = json.load(open("submissions/probes/p31_dist_wide.json")); n, ch = Counter(), Counter()
for c in S["clips"]:
    for x in c["answers"]:
        q = x["question_id"]; g = G.get(q)
        if not g or g["subcategory"] not in REC: continue
        a, w, tk = REC[g["subcategory"]]; T = TR[tk] if tk else None
        if T is not None and q not in T: continue
        ks = T[q] if T else g["probs"]
        def fuse(pv):
            s = {k: (1 - a) * g["probs"].get(k, 0) + a * pv.get(k, 0) for k in ks}
            return {k: (1 - w) * s[k] + w * T[q][k] for k in ks} if T else s
        old = fuse(V1[q]["probs"]); assert max(sorted(old), key=old.get) == x["answer"], (g["subcategory"], q)  # == p31
        pv = {k: (V1[q]["probs"].get(k, 0) + V2[q]["probs"].get(k, 0)) / 2 for k in V1[q]["probs"]}
        s = fuse(pv); new = max(sorted(s), key=s.get); n[g["subcategory"]] += 1; ch[g["subcategory"]] += new != x["answer"]; x["answer"] = new
S["meta"] = dict(S["meta"], method="p33", answer_strategy=S["meta"].get("answer_strategy", "") +
                 "; VLM probs = mean(run 1, camera run) in action_risk/decision/unsafe recipes (src/probe_vlm_avg.py)")
out = "submissions/probes/p33_vlm_avg.json"; json.dump(S, open(out, "w"))
err = validate_submission(S, data_root="cache/test")
for s in REC: print(f"  {s:32s} answered {n[s]:4d} changed vs p31 {ch[s]:3d}")
print(f"{out}: {sum(ch.values())} changed, validate errors {len(err)}, sha {hashlib.sha256(open(out, 'rb').read()).hexdigest()[:16]}")
