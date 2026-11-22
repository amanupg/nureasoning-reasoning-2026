"""2026-11-13 probe p27 (Aman approved the gray-area test-mix probe 11-13; DISCLOSE it): action_risk on the test
'Maintain speed + No lateral action' slice (189 questions, 50% of test action_risk vs 16% of validation).
p19's fusion (GBDT x VLM run 1, .5) answers Unsafe on 54/189 (29%) there; on held-out validation the same slice is
58% Unsafe (80/138) and the fusion's predicted mix matches validation gold (77 vs 80 Unsafe, acc 0.79).
p27 = p19 with the K Safe answers of that slice that lean most toward Unsafe (largest P(Unsafe) - P(Safe)) flipped to
Unsafe, K = 56, so the slice's Unsafe share equals validation's 58%. Uses only test inputs and model probabilities,
no test labels; the board result tells whether test's slice shares validation's risk mix. Note EM label-shift
estimation would push the other way (test mean P(Unsafe) 0.38 on the slice), so this is a direct hypothesis test.
Delta vs p19 0.7733 x 3000 = (truly Unsafe - truly Safe) among the 56 flipped. Stay-in-place trajectories."""
import glob, json, re, os, hashlib, sys
sys.path.insert(0, "recon/devkit")
from nureasoning.submission.format import validate_submission
PAT = re.compile(r"risk level of: '(.+?)' \(longitudinal\) \+ '(.+?)' \(lateral\)")
GT = {json.loads(l)["question_id"]: json.loads(l) for l in open("artifacts/gbdt_test_probs.jsonl")}
VT = {}
for f in glob.glob("runs_vlm/qd4b_s3000_test_s*of4.jsonl"):
    for l in open(f): r = json.loads(l); VT[r["question_id"]] = r
S = json.load(open("submissions/probes/p19_dist_vitl.json"))
cur = {a["question_id"]: a["answer"] for c in S["clips"] for a in c["answers"]}
cand, n_ms, n_uns = [], 0, 0
for c in os.listdir("cache/test"):
    p = f"cache/test/{c}/reasoning_questions.json"
    if not os.path.isfile(p): continue
    for q in json.load(open(p))["questions"]:
        if q["subcategory"] != "action_risk_assessment": continue
        lon, lat = PAT.search(q["question"]).groups()
        if not (lon == "Maintain speed" and lat == "No lateral action"): continue
        n_ms += 1; inv = {v: k for k, v in q["choices"].items()}
        g, v = GT[q["question_id"]]["probs"], VT[q["question_id"]]["probs"]
        pp = {k: .5 * g.get(k, 0) + .5 * v.get(k, 0) for k in q["choices"]}
        ans = q["choices"][cur[q["question_id"]]]
        if ans == "Unsafe": n_uns += 1
        if ans == "Safe" and "Unsafe" in inv:
            cand.append((pp[inv["Unsafe"]] - pp[inv["Safe"]], q["question_id"], inv["Unsafe"]))
K = round(80 / 138 * n_ms) - n_uns
cand.sort(reverse=True); flip = {qid: L for _, qid, L in cand[:K]}
for c in S["clips"]:
    for a in c["answers"]:
        if a["question_id"] in flip: a["answer"] = flip[a["question_id"]]
S["meta"] = dict(S["meta"], method="p27_ar_mix", answer_strategy=S["meta"].get("answer_strategy", "") +
                 f"; action_risk maintain+none slice: {K} Safe->Unsafe flips to match validation's Unsafe share (src/probe_ar_shift.py)")
out = "submissions/probes/p27_ar_mix.json"; json.dump(S, open(out, "w"))
err = validate_submission(S, data_root="cache/test")
print(f"slice {n_ms}, Unsafe before {n_uns}, K {K}, flipped margins {cand[0][0]:+.3f} .. {cand[K - 1][0]:+.3f}")
print(f"{out}: validate errors {len(err)}, sha {hashlib.sha256(open(out, 'rb').read()).hexdigest()[:16]}")
