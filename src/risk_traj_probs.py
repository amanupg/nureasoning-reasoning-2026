"""2026-11-13: action_risk probabilities from the planner's predicted future + the asked action (the decision trick:
HGB fitted on TRAIN action_risk questions with the TRUE future, applied to v9.9's predicted future on validation /
test), then fused with p19's recipe (GBDT x VLM run 1 .5) on the 864 held-out validation questions; w by cross-fit.
-> artifacts/risk_traj_probs_{validation,test}.json
usage: python src/risk_traj_probs.py <val npz> <test npz>
"""
import glob, json, os, pickle, random, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decision_traj import clip_rows
from sklearn.ensemble import HistGradientBoostingClassifier
PAT = re.compile(r"risk level of: '(.+?)' \(longitudinal\) \+ '(.+?)' \(lateral\)")
SUB = "action_risk_assessment"


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
LON = sorted({PAT.search(q["question"]).group(1) for _, q in TRQ}); LAT = sorted({PAT.search(q["question"]).group(2) for _, q in TRQ})
TR = pickle.load(open("artifacts/dec_traj_train_feats.pkl", "rb"))
for c, _ in TRQ:
    if c not in TR: TR[c] = clip_rows("train", c)


def x(f, q):
    lon, lat = PAT.search(q["question"]).groups()
    return list(f.values()) + [lon == l for l in LON] + [lat == l for l in LAT]


X = [x(TR[c], q) for c, q in TRQ if TR.get(c)]; y = [q["choices"][q["answer"]] for c, q in TRQ if TR.get(c)]
m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=0).fit(np.array(X, float), y)
print(f"train n {len(y)} classes {list(m.classes_)}", flush=True)
OUT = {}
for split, Q, P in (("validation", VAQ, pred_map(sys.argv[1])), ("test", TEQ, pred_map(sys.argv[2]))):
    F = {}; OUT[split] = {}
    for c, q in Q:
        if c not in P: continue
        if c not in F: F[c] = clip_rows(split, c, P[c])
        p = dict(zip(m.classes_, m.predict_proba(np.array([x(F[c], q)], float))[0]))
        pr = {L: p.get(t, 0.0) + 1e-6 for L, t in q["choices"].items()}; z = sum(pr.values())
        OUT[split][q["question_id"]] = {L: v / z for L, v in pr.items()}
    json.dump(OUT[split], open(f"artifacts/risk_traj_probs_{split}.json", "w"))
# validation fusion vs p19 recipe on the GBDT held-out questions
G = {r["question_id"]: r for r in map(json.loads, open("artifacts/gbdt_heldout_preds_probs.jsonl")) if r["subcategory"] == SUB}
V = {}
for f in glob.glob("runs_vlm/qd4b_s3000_val*_s*of4.jsonl"):
    if any(s in f for s in ("gray", "swap", "val150")): continue
    for l in open(f): r = json.loads(l); V[r["question_id"]] = r
Qv = {q["question_id"]: (c, q) for c, q in VAQ}; T = OUT["validation"]
ids = [i for i in G if i in V and i in T]
gold = np.array([G[i]["gold"] for i in ids]); clip = np.array([Qv[i][0] for i in ids]); log = np.array([c[:19] for c in clip])
ms = np.array([PAT.search(Qv[i][1]["question"]).groups() == ("Maintain speed", "No lateral action") for i in ids])
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
        wb = max(GRID, key=lambda w: (A[w][fit].mean(), -abs(w - .3))); cf.append(A[wb][ev].mean() - A[0.0][ev].mean())
print(f"n {len(ids)} (m+n {ms.sum()}): by w all " + " ".join(f"{w}:{A[w].mean():.3f}" for w in GRID))
print("   m+n " + " ".join(f"{w}:{A[w][ms].mean():.3f}" for w in GRID))
print(f"traj-only acc {A[1.0].mean():.3f}; cross-fit gain {np.mean(cf):+.4f} (pos share {np.mean(np.array(cf) > 0):.2f})")
