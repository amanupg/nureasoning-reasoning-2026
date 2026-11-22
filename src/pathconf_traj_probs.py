"""2026-11-13: future_path_intersection / conflict_prediction from the planner's predicted ego future + the stated
object position and class (decision trick). HGB on TRAIN questions with the TRUE ego future, applied to v9.9's
predicted future; fused with p19's answers' probabilities (GBDT held-out) on validation, w by cross-fit.
-> artifacts/pathconf_traj_probs_{validation,test}.json
usage: python src/pathconf_traj_probs.py <val npz> <test npz>
"""
import glob, json, os, pickle, random, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decision_traj import clip_rows
from sklearn.ensemble import HistGradientBoostingClassifier
SUBS = ("future_path_intersection", "conflict_prediction")
RX = re.compile(r"For the (.+?) in the (.+?) camera .*?\[x, y\] = \[(-?[\d.]+), (-?[\d.]+)\]")
CATS = ["car", "human", "bicycle", "traffic_cone", "door", "construction_zone_area", "animal"]


def qs(split):
    out = []
    if split == "test":
        for c in sorted(os.listdir("cache/test")):
            p = f"cache/test/{c}/reasoning_questions.json"
            if os.path.isfile(p): out += [(c, q) for q in json.load(open(p))["questions"] if q["subcategory"] in SUBS]
    else:
        for f in glob.glob(f"{'vqa_train' if split == 'train' else 'vqa_val'}/*/*_vqa.json"):
            out += [(f.split("/")[1], q) for q in json.load(open(f))["questions"] if q["subcategory"] in SUBS]
    return out


def pred_map(path):
    z = np.load(path, allow_pickle=True); return {str(n): t for n, t in zip(z["names"], z["traj"])}


def x(f, q):
    m = RX.search(q["question"]); cat, cam, ox, oy = m.group(1), m.group(2), float(m.group(3)), float(m.group(4))
    r = np.hypot(ox, oy)
    # ego future vs object position: lateral offset of the object from the ego's predicted endpoint path
    return list(f.values()) + [ox, oy, r, ox / (r + 1e-3), oy / (r + 1e-3), ox - f["x50"], oy - f["y50"], ox - f["x20"], oy - f["y20"]] + [cat == k for k in CATS]


def yes(q, L): return q["choices"][L].lower().startswith("yes")


TRQ, VAQ, TEQ = qs("train"), qs("validation"), qs("test")
TR = pickle.load(open("artifacts/dec_traj_train_feats.pkl", "rb"))
for c, _ in TRQ:
    if c not in TR: TR[c] = clip_rows("train", c)
OUT = {"validation": {}, "test": {}}
PV, PT = pred_map(sys.argv[1]), pred_map(sys.argv[2])
F = {"validation": {}, "test": {}}
for sub in SUBS:
    rows = [(c, q) for c, q in TRQ if q["subcategory"] == sub and TR.get(c) and RX.search(q["question"])]
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=0).fit(
        np.array([x(TR[c], q) for c, q in rows], float), [int(yes(q, q["answer"])) for c, q in rows])
    for sp, Q, P in (("validation", VAQ, PV), ("test", TEQ, PT)):
        for c, q in Q:
            if q["subcategory"] != sub or c not in P or not RX.search(q["question"]): continue
            if c not in F[sp]: F[sp][c] = clip_rows(sp, c, P[c])
            py = float(m.predict_proba(np.array([x(F[sp][c], q)], float))[0, 1])
            OUT[sp][q["question_id"]] = {L: (py if yes(q, L) else 1 - py) for L in q["choices"]}
    print(f"{sub}: train {len(rows)}, yes share {np.mean([yes(q, q['answer']) for c, q in rows]):.3f}", flush=True)
for sp in OUT: json.dump(OUT[sp], open(f"artifacts/pathconf_traj_probs_{sp}.json", "w"))
G = {r["question_id"]: r for r in map(json.loads, open("artifacts/gbdt_heldout_preds_probs.jsonl")) if r["subcategory"] in SUBS}
Qv = {q["question_id"]: (c, q) for c, q in VAQ}; T = OUT["validation"]
GRID = [round(v, 1) for v in np.arange(0, 1.01, 0.1)]
for sub in SUBS:
    allv = [i for i, (c, q) in Qv.items() if q["subcategory"] == sub and i in T]
    print(f"{sub}: traj-only acc all val {np.mean([max(T[i], key=T[i].get) == Qv[i][1]['answer'] for i in allv]):.3f} (n {len(allv)})")
    ids = [i for i in G if G[i]["subcategory"] == sub and i in T]
    if not ids: continue
    gold = np.array([G[i]["gold"] for i in ids]); log = np.array([Qv[i][0][:19] for i in ids])
    A = {}
    for w in GRID:
        A[w] = np.array([max(sorted(T[i]), key=lambda k: (1 - w) * G[i]["probs"].get(k, 0) + w * T[i][k]) == g for i, g in zip(ids, gold)])
    logs = sorted(set(log)); rng = random.Random(0); cf = []
    for _ in range(20):
        rng.shuffle(logs); h = set(logs[: len(logs) // 2]); mA = np.array([l in h for l in log])
        for fit, ev in ((mA, ~mA), (~mA, mA)):
            wb = max(GRID, key=lambda w: (A[w][fit].mean(), -abs(w - .5))); cf.append(A[wb][ev].mean() - A[0.0][ev].mean())
    print(f"  held-out n {len(ids)} by w " + " ".join(f"{w}:{A[w].mean():.3f}" for w in GRID))
    print(f"  cross-fit {np.mean(cf):+.4f} (pos {np.mean(np.array(cf) > 0):.2f}); w .5 wins {int((A[.5] > A[0.0]).sum())} losses {int((A[.5] < A[0.0]).sum())}")
