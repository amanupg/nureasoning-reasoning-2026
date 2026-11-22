"""2026-11-15: p33 candidate = deployed recipes with VLM run 1's probabilities replaced by the mean of run 1 and the
camera run (qd4b_r1fe), same (a, w). Held-out validation, paired vs deployed, clip bootstrap. Types: action_risk
(GBDT x VLM .5), joint/lateral (a .4/.5, then 50/50 planner), unsafe (a .5, then 50/50 planner). Longitudinal uses
a = 0 (no VLM), so it is unchanged.
usage: python src/p33_avg_eval.py"""
import glob, json, numpy as np
SKIP = ("gray", "swap", "val150", "_p1_", "_p2_", "_p3_", "pavg", "valfus")
def load(pat):
    D = {}
    for f in glob.glob(pat):
        if any(s in f for s in SKIP): continue
        for l in open(f): r = json.loads(l); D[r["question_id"]] = r
    return D
G = {r["question_id"]: r for r in map(json.loads, open("artifacts/gbdt_heldout_preds_probs.jsonl"))}
V1, V2 = load("runs_vlm/qd4b_s3000_val*_s*of4.jsonl"), load("runs_vlm/qd4b_r1fe/val_s*of2.jsonl")
TR = {"dec": json.load(open("artifacts/dec_traj_probs_validation.json")), "unsafe": json.load(open("artifacts/unsafe_traj_probs_validation.json"))}
DEP = {"action_risk_assessment": (.5, 0, None), "driving_decision_joint": (.4, .5, "dec"),
       "driving_decision_lateral": (.5, .5, "dec"), "unsafe_action_identification": (.5, .5, "unsafe")}
tot = []
for sub, (a, w, tk) in DEP.items():
    T = TR[tk] if tk else None
    qs = sorted(q for q in G if G[q]["subcategory"] == sub and q in V1 and q in V2 and G[q]["gold"] is not None and (T is None or q in T))
    def acc(avg):
        r = []
        for q in qs:
            pv = {k: (V1[q]["probs"].get(k, 0) + V2[q]["probs"].get(k, 0)) / 2 for k in V1[q]["probs"]} if avg else V1[q]["probs"]
            ks = T[q] if T else G[q]["probs"]
            s = {k: (1 - a) * G[q]["probs"].get(k, 0) + a * pv.get(k, 0) for k in ks}
            if T: s = {k: (1 - w) * s[k] + w * T[q][k] for k in ks}
            r.append(max(sorted(s), key=s.get) == G[q]["gold"])
        return np.array(r, float)
    dep, new = acc(False), acc(True); d = new - dep; clip = np.array([V1[q]["clip"] for q in qs])
    uc = sorted(set(clip)); idx = {c: np.where(clip == c)[0] for c in uc}; br = np.random.default_rng(0)
    bs = [np.concatenate([d[idx[c]] for c in br.choice(uc, len(uc))]).sum() for _ in range(2000)]
    print(f"{sub[:30]:30s} n {len(qs)} deployed {dep.mean():.3f} avg {new.mean():.3f} net {int(d.sum()):+d} "
          f"[{np.percentile(bs, 2.5):+.0f}, {np.percentile(bs, 97.5):+.0f}] wins {int((d > 0).sum())} losses {int((d < 0).sum())}")
