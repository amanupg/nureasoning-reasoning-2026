"""2026-11-15: reasoning-side probabilities for the planning thread (Aman: work hand in hand). One jsonl per split under
artifacts/for_planning/, one row per question of: driving_decision_{joint,lateral,longitudinal}, action_risk_assessment,
unsafe_action_identification, distance_from_2d_bbox.
Fields: split, question_id, clip, subcategory, question, choices, target_track_token, gold (validation only),
  p_gbdt   GBDT class probs. validation: held out by drive-log half (fit on train + the other half), never saw the row.
           test: fit on train + validation.
  p_vlm    VLM run 1 (Qwen-Drive 4B LoRA, train only) letter probs (validation: val + valfus runs), null if absent.
  p_fused  deployed GBDT x VLM mix WITHOUT any planner input: joint a .4, lateral .5, longitudinal 0 (GBDT only),
           action_risk .5, unsafe .5. Use this one as a planner input (no circularity).
  p_deployed  what the submission uses: p_fused 50/50 with probs from v9.9's predicted trajectory for the decision
           and unsafe types (circular if fed back into a planner), = p_fused for action_risk.
  dist_m   distance only: geometric mean of the ViT-H+ distance nets (validation: 3 train-only nets ens_h16_s0-2;
           test: the deployed 6 incl. train+val nets). Choices are distances in metres for the target object's 2D box.
usage: python src/export_for_planning.py"""
import glob, json, math, os, numpy as np
SUBS = {"driving_decision_joint": .4, "driving_decision_lateral": .5, "driving_decision_longitudinal": 0.0,
        "action_risk_assessment": .5, "unsafe_action_identification": .5, "distance_from_2d_bbox": None}
TRK = {"driving_decision_joint": "dec", "driving_decision_lateral": "dec", "driving_decision_longitudinal": "dec",
       "unsafe_action_identification": "unsafe"}
SKIP = ("gray", "swap", "val150", "_p1_", "_p2_", "_p3_", "pavg")   # keep valfus: run 1 on the fusion questions
def vload(pat):
    D = {}
    for f in glob.glob(pat):
        if any(s in f for s in SKIP): continue
        for l in open(f): r = json.loads(l); D[r["question_id"]] = r["probs"]
    return D
def gm(nets, split):
    ds = [json.load(open(f"runs_dist/{n}/pred_{split}.json")) for n in nets]
    return {q: math.exp(np.mean([math.log(d[q]) for d in ds])) for q in ds[0]}
os.makedirs("artifacts/for_planning", exist_ok=True)
for split in ("validation", "test"):
    if split == "validation":
        G = {r["question_id"]: r for r in map(json.loads, open("artifacts/gbdt_heldout_preds_probs.jsonl"))}
        Q = {q["question_id"]: dict(q, _clip=f.split("/")[1]) for f in glob.glob("vqa_val/*/*_vqa.json") for q in json.load(open(f)).get("questions") or []}
        V = vload("runs_vlm/qd4b_s3000_val*_s*of4.jsonl"); D = gm([f"ens_h16_s{i}" for i in range(3)], "validation")
    else:
        G = {r["question_id"]: r for r in map(json.loads, open("artifacts/gbdt_test_probs.jsonl"))}
        Q = {q["question_id"]: dict(q, _clip=f.split("/")[2]) for f in glob.glob("cache/test/*/reasoning_questions.json") for q in json.load(open(f))["questions"]}
        V = vload("runs_vlm/qd4b_s3000_test_s*of4.jsonl")
        D = gm([f"ens_h16_s{i}" for i in range(3)] + [f"ens_h16_tv_s{i}" for i in range(3)], "test")
    T = {k: json.load(open(f"artifacts/{k}_traj_probs_{split}.json")) for k in ("dec", "unsafe")}
    n = {}; out = open(f"artifacts/for_planning/{split}_reasoning_probs.jsonl", "w")
    for qid, q in Q.items():
        s = q["subcategory"]
        if s not in SUBS: continue
        g = G.get(qid); clip = q["_clip"]
        r = dict(split=split, question_id=qid, clip=clip, subcategory=s, question=q["question"], choices=q["choices"],
                 target_track_token=q.get("target_track_token"), gold=q.get("answer") if split == "validation" else None)
        if s == "distance_from_2d_bbox":
            r["dist_m"] = D.get(qid)
        else:
            if g is None: continue
            a = SUBS[s]; pv = V.get(qid); pg = g["probs"]; ks = sorted(q["choices"])
            pf = {k: (1 - a) * pg.get(k, 0) + a * (pv or pg).get(k, 0) for k in ks}
            pt = T[TRK[s]].get(qid) if s in TRK else None
            r.update(p_gbdt=pg, p_vlm=pv, p_fused=pf, p_deployed={k: .5 * pf[k] + .5 * pt[k] for k in ks} if pt else pf)
        out.write(json.dumps(r) + "\n"); n[s] = n.get(s, 0) + 1
    out.close(); print(split, n)
