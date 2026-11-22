"""2026-11-17 action_risk inside the deployed recipe: p = .5 GBDT(held-out) + .5 VLM; run 1 vs a new VLM, same questions,
paired clip-bootstrap CI; also the new VLM alone and the GBDT alone. usage: python src/vlm_fuse_ar_eval.py <new val glob>"""
import glob, json, sys, numpy as np
SKIP = ("gray", "swap", "val150", "_p1_", "_p2_", "_p3_", "pavg", "valfus")
def load(p):
    D = {}
    for f in glob.glob(p):
        if any(s in f for s in SKIP): continue
        for l in open(f): r = json.loads(l); D[r["question_id"]] = r
    return D
G = {r["question_id"]: r for r in map(json.loads, open("artifacts/gbdt_heldout_preds_probs.jsonl"))}
V1, V2 = load("runs_vlm/qd4b_s3000_val*_s*of4.jsonl"), load(sys.argv[1])
for sub in sys.argv[2:] or ["action_risk_assessment"]:
    qs = sorted(q for q in G if G[q]["subcategory"] == sub and q in V1 and q in V2 and G[q]["gold"] is not None and G[q]["probs"])
    if not qs: print(sub, "no questions with GBDT probs"); continue
    gold = np.array([G[q]["gold"] for q in qs]); clip = np.array([V2[q]["clip"] for q in qs])
    def acc(V, a): return np.array([max(sorted(G[q]["probs"]), key=lambda k: (1 - a) * G[q]["probs"].get(k, 0) + a * V[q]["probs"].get(k, 0)) == g for q, g in zip(qs, gold)], float)
    d1, d2 = acc(V1, .5), acc(V2, .5); d = d2 - d1
    uc = sorted(set(clip)); idx = {c: np.where(clip == c)[0] for c in uc}; br = np.random.default_rng(0)
    bs = [np.concatenate([d[idx[c]] for c in br.choice(uc, len(uc))]).mean() for _ in range(2000)]
    print(f"{sub[:30]:30s} n {len(qs)} GBDT {acc(V1, 0).mean():.3f} | run1 alone {acc(V1, 1).mean():.3f} new alone {acc(V2, 1).mean():.3f} | "
          f"fused .5: run1 {d1.mean():.3f} new {d2.mean():.3f} diff {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] wins {int((d > 0).sum())} losses {int((d < 0).sum())}")
