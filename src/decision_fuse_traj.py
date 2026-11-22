"""Validation: driving-decision answers = deployed recipe (GBDT x VLM run 1 fusion, alphas joint .4 / lateral .5 /
longitudinal GBDT) vs + planner-trajectory probabilities (artifacts/dec_traj_probs_validation.json).
p = (1 - w) * p_deployed + w * p_traj; w chosen on one half of the drive logs, scored on the other (20 splits x 2).
Paired clip-bootstrap CI for the chosen-on-all w and for traj alone. Same questions for every variant.
"""
import glob, json, random, numpy as np
AL = {"driving_decision_joint": .4, "driving_decision_lateral": .5, "driving_decision_longitudinal": 0.0}
G = {r["question_id"]: r for r in map(json.loads, open("artifacts/gbdt_heldout_preds_probs.jsonl"))}
V = {}
for f in glob.glob("runs_vlm/qd4b_s3000_val*_s*of4.jsonl"):
    if "gray" in f or "swap" in f or "val150" in f: continue
    for l in open(f): r = json.loads(l); V[r["question_id"]] = r
import sys
T = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "artifacts/dec_traj_probs_validation.json"))
GRID = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
res = {}
for sub, al in AL.items():
    qs = [q for q in T if q in G and G[q].get("subcategory", sub) == sub and (al == 0 or q in V)]
    qs = [q for q in qs if G[q]["gold"] is not None]
    if not qs: continue
    gold = np.array([G[q]["gold"] for q in qs]); clip = np.array([V[q]["clip"] if q in V else G[q].get("clip", q) for q in qs])
    log = np.array([c[:19] for c in clip])
    def acc(w):
        out = []
        for q, g in zip(qs, gold):
            pg = G[q]["probs"]; pv = V[q]["probs"] if q in V else pg; pt = T[q]
            d = {k: (1 - al) * pg.get(k, 0) + al * pv.get(k, 0) for k in pt}
            s = {k: (1 - w) * d[k] + w * pt[k] for k in pt}
            out.append(max(sorted(s), key=s.get) == g)
        return np.array(out)
    A = {w: acc(w) for w in GRID}
    logs = sorted(set(log)); rng = random.Random(0); cf = []
    for _ in range(20):
        rng.shuffle(logs); h = set(logs[: len(logs) // 2]); mA = np.array([l in h for l in log])
        for fit, ev in ((mA, ~mA), (~mA, mA)):
            wb = max(GRID, key=lambda w: (A[w][fit].mean(), -abs(w - .5)))
            cf.append(A[wb][ev].mean() - A[0.0][ev].mean())
    wall = max(GRID, key=lambda w: (A[w].mean(), -abs(w - .5)))
    uc = sorted(set(clip)); idx = {c: np.where(clip == c)[0] for c in uc}; brng = np.random.default_rng(0)
    def ci(d): bs = [np.concatenate([d[idx[c]] for c in brng.choice(uc, len(uc))]).mean() for _ in range(2000)]; return np.percentile(bs, [2.5, 97.5])
    for w in (wall, 1.0):
        d = A[w].astype(float) - A[0.0]; lo, hi = ci(d)
        print(f"{sub:30s} n {len(qs)} deployed {A[0.0].mean():.3f} | w {w} {A[w].mean():.3f} diff {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] (wins {int((d > 0).sum())} losses {int((d < 0).sum())})")
    print(f"{'':30s} crossfit gain {np.mean(cf):+.4f} (pos share {np.mean(np.array(cf) > 0):.2f}); by w: " + " ".join(f"{w}:{A[w].mean():.3f}" for w in GRID))
    res[sub] = {"w_all": wall, "crossfit": float(np.mean(cf))}
if len(sys.argv) == 1: json.dump(res, open("artifacts/dec_traj_fuse_val.json", "w"), indent=1)
