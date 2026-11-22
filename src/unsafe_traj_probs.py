"""2026-11-13: unsafe_action_identification from the planner's predicted future (decision trick). Per-OPTION HGB on
TRAIN questions with the TRUE future: features = future kinematics + the option's (longitudinal, lateral) one-hots,
label = option is the unsafe one; per question the option scores are normalised. Applied to v9.9's predicted future;
fused with p19's recipe (GBDT x VLM run 1 .5) on held-out validation, w by cross-fit.
-> artifacts/unsafe_traj_probs_{validation,test}.json
usage: python src/unsafe_traj_probs.py <val npz> <test npz>
"""
import glob, json, os, pickle, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decision_traj import clip_rows
from sklearn.ensemble import HistGradientBoostingClassifier
SUB = "unsafe_action_identification"


def qs(split):
    out = []
    if split == "test":
        for c in sorted(os.listdir("cache/test")):
            p = f"cache/test/{c}/reasoning_questions.json"
            if os.path.isfile(p): out += [(c, q) for q in json.load(open(p))["questions"] if q["subcategory"] == SUB]
    else:
        for f in glob.glob(f"{'vqa_train' if split == 'train' else 'vqa_val'}/*/*_vqa.json"):
            out += [(f.split("/")[1], q) for q in json.load(open(f))["questions"] if q["subcategory"] == SUB]
    return out


def pred_map(path):
    z = np.load(path, allow_pickle=True); return {str(n): t for n, t in zip(z["names"], z["traj"])}


TRQ, VAQ, TEQ = qs("train"), qs("validation"), qs("test")
split = lambda t: [s.strip() for s in t.split("+")]
LON = sorted({split(v)[0] for _, q in TRQ for v in q["choices"].values()}); LAT = sorted({split(v)[1] for _, q in TRQ for v in q["choices"].values()})
TR = pickle.load(open("artifacts/dec_traj_train_feats.pkl", "rb"))
for c, _ in TRQ:
    if c not in TR: TR[c] = clip_rows("train", c)
x = lambda f, t: list(f.values()) + [split(t)[0] == l for l in LON] + [split(t)[1] == l for l in LAT]
X, y = [], []
for c, q in TRQ:
    if not TR.get(c): continue
    for L, t in q["choices"].items(): X.append(x(TR[c], t)); y.append(int(L == q["answer"]))
m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=0).fit(np.array(X, float), y)
print(f"train options {len(y)} ({len(TRQ)} questions)", flush=True)
OUT = {}
for sp, Q, P in (("validation", VAQ, pred_map(sys.argv[1])), ("test", TEQ, pred_map(sys.argv[2]))):
    F = {}; OUT[sp] = {}
    for c, q in Q:
        if c not in P: continue
        if c not in F: F[c] = clip_rows(sp, c, P[c])
        Ls = sorted(q["choices"]); s = m.predict_proba(np.array([x(F[c], q["choices"][L]) for L in Ls], float))[:, 1] + 1e-6
        OUT[sp][q["question_id"]] = dict(zip(Ls, (s / s.sum()).tolist()))
    json.dump(OUT[sp], open(f"artifacts/unsafe_traj_probs_{sp}.json", "w"))
G = {r["question_id"]: r for r in map(json.loads, open("artifacts/gbdt_heldout_preds_probs.jsonl")) if r["subcategory"] == SUB}
V = {}
for f in glob.glob("runs_vlm/qd4b_s3000_val*_s*of4.jsonl"):
    if any(k in f for k in ("gray", "swap", "val150")): continue
    for l in open(f): r = json.loads(l); V[r["question_id"]] = r
Qv = {q["question_id"]: c for c, q in VAQ}; T = OUT["validation"]
allv = [q["question_id"] for _, q in VAQ if q["question_id"] in T]
gold_all = {q["question_id"]: q["answer"] for _, q in VAQ}
print(f"traj-only acc on all {len(allv)} val: {np.mean([max(T[i], key=T[i].get) == gold_all[i] for i in allv]):.3f}")
ids = [i for i in G if i in V and i in T]
gold = np.array([G[i]["gold"] for i in ids]); log = np.array([Qv[i][:19] for i in ids])
GRID = [round(v, 1) for v in np.arange(0, 1.01, 0.1)]
def acc(w):
    o = []
    for i, g in zip(ids, gold):
        d = {k: .5 * G[i]["probs"].get(k, 0) + .5 * V[i]["probs"].get(k, 0) for k in T[i]}
        s = {k: (1 - w) * d[k] + w * T[i][k] for k in T[i]}; o.append(max(sorted(s), key=s.get) == g)
    return np.array(o)
A = {w: acc(w) for w in GRID}
logs = sorted(set(log)); rng = random.Random(0); cf = []
for _ in range(20):
    rng.shuffle(logs); h = set(logs[: len(logs) // 2]); mA = np.array([l in h for l in log])
    for fit, ev in ((mA, ~mA), (~mA, mA)):
        wb = max(GRID, key=lambda w: (A[w][fit].mean(), -abs(w - .5))); cf.append(A[wb][ev].mean() - A[0.0][ev].mean())
print(f"held-out n {len(ids)}: by w " + " ".join(f"{w}:{A[w].mean():.3f}" for w in GRID))
print(f"cross-fit gain {np.mean(cf):+.4f} (pos share {np.mean(np.array(cf) > 0):.2f})")
d = A[0.5].astype(float) - A[0.0]; print(f"w .5 wins {int((d > 0).sum())} losses {int((d < 0).sum())}")
