"""2026-11-17 validation: does a new decision-probability source X help on top of what is deployed (p30)?
Deployed decisions: d = GBDT x VLM run 1 (alphas joint .4 / lateral .5 / longitudinal 0), then p30 = .5 d + .5 traj
(artifacts/dec_traj_probs_validation.json). Variants, same questions: X alone; p30 -> (1-w) p30 + w X.
w chosen on half the drive logs, scored on the other half (20 splits x 2) = cross-fit gain; clip-bootstrap CI.
usage: python src/decision_fuse3.py <X validation probs json>"""
import glob, json, random, sys, numpy as np
AL = {"driving_decision_joint": .4, "driving_decision_lateral": .5, "driving_decision_longitudinal": 0.0}
G = {r["question_id"]: r for r in map(json.loads, open("artifacts/gbdt_heldout_preds_probs.jsonl"))}
V = {}
for f in glob.glob("runs_vlm/qd4b_s3000_val*_s*of4.jsonl"):
    if "gray" in f or "swap" in f or "val150" in f: continue
    for l in open(f): r = json.loads(l); V[r["question_id"]] = r
T = json.load(open("artifacts/dec_traj_probs_validation.json")); X = json.load(open(sys.argv[1]))
GRID = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
for sub, al in AL.items():
    qs = [q for q in T if q in X and q in G and G[q].get("subcategory", sub) == sub and (al == 0 or q in V) and G[q]["gold"] is not None]
    gold = np.array([G[q]["gold"] for q in qs]); clip = np.array([V[q]["clip"] if q in V else G[q].get("clip", q) for q in qs])
    log = np.array([c[:19] for c in clip])
    def p30(q):
        pg = G[q]["probs"]; pv = V[q]["probs"] if q in V else pg; pt = T[q]
        return {k: .5 * ((1 - al) * pg.get(k, 0) + al * pv.get(k, 0)) + .5 * pt[k] for k in pt}
    def acc(w):
        out = []
        for q, g in zip(qs, gold):
            b = p30(q); s = {k: (1 - w) * b[k] + w * X[q].get(k, 0) for k in b}
            out.append(max(sorted(s), key=s.get) == g)
        return np.array(out)
    A = {w: acc(w) for w in GRID}
    logs = sorted(set(log)); rng = random.Random(0); cf = []
    for _ in range(20):
        rng.shuffle(logs); h = set(logs[: len(logs) // 2]); m = np.array([l in h for l in log])
        for fit, ev in ((m, ~m), (~m, m)):
            wb = max(GRID, key=lambda w: (A[w][fit].mean(), -w))
            cf.append(A[wb][ev].mean() - A[0.0][ev].mean())
    uc = sorted(set(clip)); idx = {c: np.where(clip == c)[0] for c in uc}; br = np.random.default_rng(0)
    def ci(d): bs = [np.concatenate([d[idx[c]] for c in br.choice(uc, len(uc))]).mean() for _ in range(2000)]; return np.percentile(bs, [2.5, 97.5])
    wall = max(GRID, key=lambda w: (A[w].mean(), -w))
    print(f"{sub:30s} n {len(qs)} p30 {A[0.0].mean():.3f} | X alone {A[1.0].mean():.3f}")
    for w in (wall, 1.0):
        d = A[w].astype(float) - A[0.0]; lo, hi = ci(d)
        print(f"{'':30s} w {w}: {A[w].mean():.3f} diff {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] wins {int((d > 0).sum())} losses {int((d < 0).sum())}")
    print(f"{'':30s} crossfit gain {np.mean(cf):+.4f} (pos share {np.mean(np.array(cf) > 0):.2f}); by w: " + " ".join(f"{w}:{A[w].mean():.3f}" for w in GRID))
