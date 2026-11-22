"""2026-11-14: does the new VLM run (qd4b_r1fe: Qwen-Drive 4B from run 1 + vision LoRA + FOCUS cams + EGO line) improve
the DEPLOYED decision / unsafe recipes when it replaces run 1? Held-out validation only (GBDT held-out probs,
artifacts/gbdt_heldout_preds_probs.jsonl; both VLMs train-only; planner-future probs from train-fitted HGBs on Edge's
predicted futures, Edge never saw validation). Recipe: p = (1-w) [(1-a) p_GBDT + a p_VLM] + w p_traj.
deployed = run 1 with the deployed (a, w); 'same' = new VLM with the deployed (a, w) (for longitudinal a .5, since
deployed a 0 ignores the VLM); 'cf' = (a, w) chosen on half the drive logs, scored on the other half (20 x 2).
usage: python src/vlm2_fuse_eval.py [new run val glob]"""
import glob, json, random, sys, numpy as np
NEW = sys.argv[1] if len(sys.argv) > 1 else "runs_vlm/qd4b_r1fe/val_s*of2.jsonl"
DEP = {"driving_decision_joint": (.4, .5, "dec"), "driving_decision_lateral": (.5, .5, "dec"),
       "driving_decision_longitudinal": (0.0, .5, "dec"), "unsafe_action_identification": (.5, .5, "unsafe")}
SKIP = ("gray", "swap", "val150", "_p1_", "_p2_", "_p3_", "pavg", "valfus")
def load(pat):
    D = {}
    for f in glob.glob(pat):
        if any(s in f for s in SKIP): continue
        for l in open(f): r = json.loads(l); D[r["question_id"]] = r
    return D
G = {r["question_id"]: r for r in map(json.loads, open("artifacts/gbdt_heldout_preds_probs.jsonl"))}
V1, V2 = load("runs_vlm/qd4b_s3000_val*_s*of4.jsonl"), load(NEW)
TR = {"dec": json.load(open("artifacts/dec_traj_probs_validation.json")), "unsafe": json.load(open("artifacts/unsafe_traj_probs_validation.json"))}
GRID = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]; out = {}
for sub, (a0, w0, tk) in DEP.items():
    T = TR[tk]
    qs = sorted(q for q in T if q in G and G[q]["subcategory"] == sub and q in V1 and q in V2 and G[q]["gold"] is not None)
    gold = np.array([G[q]["gold"] for q in qs]); clip = np.array([V2[q]["clip"] for q in qs]); log = np.array([c[:19] for c in clip])
    def acc(V, a, w):
        r = []
        for q, g in zip(qs, gold):
            pt = T[q]; s = {k: (1 - w) * ((1 - a) * G[q]["probs"].get(k, 0) + a * V[q]["probs"].get(k, 0)) + w * pt[k] for k in pt}
            r.append(max(sorted(s), key=s.get) == g)
        return np.array(r, float)
    dep = acc(V1, a0, w0); same = acc(V2, a0 if a0 > 0 else .5, w0)
    A = {(a, w): acc(V2, a, w) for a in GRID for w in GRID}
    logs = sorted(set(log)); rng = random.Random(0); cf = []
    for _ in range(20):
        rng.shuffle(logs); h = set(logs[: len(logs) // 2]); mA = np.array([l in h for l in log])
        for fit, ev in ((mA, ~mA), (~mA, mA)):
            ab = max(A, key=lambda k: (A[k][fit].mean(), -abs(k[0] - .5) - abs(k[1] - .5))); cf.append(A[ab][ev].mean() - dep[ev].mean())
    best = max(A, key=lambda k: (A[k].mean(), -abs(k[0] - .5) - abs(k[1] - .5)))
    uc = sorted(set(clip)); idx = {c: np.where(clip == c)[0] for c in uc}; br = np.random.default_rng(0)
    d = same - dep; bs = [np.concatenate([d[idx[c]] for c in br.choice(uc, len(uc))]).mean() for _ in range(2000)]
    print(f"{sub[:30]:30s} n {len(qs)} deployed(run1 a{a0} w{w0}) {dep.mean():.3f} | new same-weights {same.mean():.3f} "
          f"diff {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] wins {int((d > 0).sum())} losses {int((d < 0).sum())}"
          f" | best (a,w) {best} {A[best].mean():.3f} | cross-fit gain vs deployed {np.mean(cf):+.4f} (pos {np.mean(np.array(cf) > 0):.2f})", flush=True)
    out[sub] = dict(n=len(qs), deployed=dep.mean(), same=same.mean(), best=best, crossfit=float(np.mean(cf)))
json.dump(out, open("artifacts/vlm2_fuse_val.json", "w"), indent=1)
