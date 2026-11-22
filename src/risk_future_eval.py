"""2026-11-11: does the planning blend's predicted future (v9.4 recipe, jepa runs/handoff/v94_blend_validation_traj.json,
read-only) add to our action_risk answers? Validation, the GBDT held-out half (gbdt_heldout_preds_probs.jsonl, 864 q):
baseline = GBDT answer; candidate = HGB on [GBDT probs + predicted-future kinematics + action one-hots], 5-fold
GroupKFold by log. Paired difference with a clip-cluster bootstrap CI. Also the 'Maintain speed' subset (half of test)."""
import glob, json, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
T = json.load(open("../jepa-nureasoning/runs/handoff/v94_blend_validation_traj.json"))["clips"]
Q = {}
for f in glob.glob("vqa_val/*/*_vqa.json"):
    c = f.split("/")[1]
    for q in json.load(open(f))["questions"]:
        if q["subcategory"] == "action_risk_assessment": q["clip"] = c; Q[q["question_id"]] = q
G = {r["question_id"]: r for r in map(json.loads, open("artifacts/gbdt_heldout_preds_probs.jsonl")) if r["question_id"] in Q}
LON = sorted({q["longitudinal"] for q in Q.values()}); LAT = sorted({q["lateral"] for q in Q.values()})
LABS = ["Safe", "Unsafe", "Suboptimal"]


def kin(clip):
    p = np.array(T[clip]["ego_xyyaw"]); v = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1) / 0.1   # 50 speeds
    vs = [v[:5].mean(), v[5:10].mean(), v[15:20].mean(), v[25:30].mean(), v[45:50].mean()]
    return vs + [vs[2] - vs[0], vs[4] - vs[0], v.min(), p[-1, 1], p[-1, 2], p[-1, 0]]


rows = []
for qid, g in G.items():
    q = Q[qid]
    if q["clip"] not in T: continue
    gold = q["choices"][q["answer"]]
    inv = {v: k for k, v in q["choices"].items()}
    gp = [g["probs"].get(inv.get(L, "?"), 0.0) for L in LABS]
    x = gp + kin(q["clip"]) + [q["longitudinal"] == l for l in LON] + [q["lateral"] == l for l in LAT]
    rows.append((x, LABS.index(gold) if gold in LABS else -1, q["choices"][g["model"]] == gold, q["clip"], q["clip"][:19],
                 q["longitudinal"] == "Maintain speed"))
X = np.array([r[0] for r in rows], float); y = np.array([r[1] for r in rows]); base = np.array([r[2] for r in rows])
clip = np.array([r[3] for r in rows]); grp = np.array([r[4] for r in rows]); ms = np.array([r[5] for r in rows])
print("n", len(y), "maintain speed", ms.sum(), "labels", np.bincount(y + 1))
cand = np.zeros(len(y), bool)
for tr, te in GroupKFold(5).split(X, y, grp):
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0).fit(X[tr], y[tr])
    cand[te] = m.predict(X[te]) == y[te]
for nm, mk in (("all", np.ones(len(y), bool)), ("maintain speed", ms)):
    d = cand[mk].astype(float) - base[mk]; cl = clip[mk]; uc = np.unique(cl); rng = np.random.default_rng(0)
    idx = {c: np.where(cl == c)[0] for c in uc}; bs = []
    for _ in range(2000):
        s = rng.choice(uc, len(uc)); bs.append(np.concatenate([d[idx[c]] for c in s]).mean())
    print(f"{nm:15s} n {mk.sum():4d}: GBDT {base[mk].mean():.3f}  +predicted future {cand[mk].mean():.3f}  "
          f"diff {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
# control: same model without the trajectory features
cand2 = np.zeros(len(y), bool); keep = [i for i in range(X.shape[1]) if not (3 <= i < 3 + 11)]
for tr, te in GroupKFold(5).split(X, y, grp):
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0).fit(X[tr][:, keep], y[tr])
    cand2[te] = m.predict(X[te][:, keep]) == y[te]
print(f"control (GBDT probs + actions only, same refit): all {cand2.mean():.3f}, maintain speed {cand2[ms].mean():.3f}")
for nm, mk in (("all", np.ones(len(y), bool)), ("maintain speed", ms)):
    d = cand[mk].astype(float) - cand2[mk]; cl = clip[mk]; uc = np.unique(cl); rng = np.random.default_rng(1)
    idx = {c: np.where(cl == c)[0] for c in uc}
    bs = [np.concatenate([d[idx[c]] for c in rng.choice(uc, len(uc))]).mean() for _ in range(2000)]
    print(f"trajectory features vs control, {nm}: {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
