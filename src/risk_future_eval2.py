"""2026-11-12: action_risk labels are Gemini judgements conditioned on the ground-truth future, and half of test's
action_risk questions are 'Maintain speed + No lateral action' (16% on validation; GBDT 0.717 there vs 0.883 on
other actions). Does the future help, and does the new Edge planner (v9.7, planning 0.8091 on test) carry that
signal better than the v9.4 blend that p21 used (p21 lost 9 answers vs p20)?
Validation, GBDT held-out half (864 q), 5-fold GroupKFold by log, HGB on [GBDT probs + action one-hots (+ future)]:
  control   : no future
  oracle    : human future (jepa data/edge_inputs_validation_dupfix.json fut64) -> ceiling for any planner
  edge3215  : Edge ft3215 predicted future (jepa runs/edge_ft3215_dupfix_validation.npz = v9.7's model)
  v94       : v9.4 blend predicted future (p21's input)
All jepa files read-only. Paired clip-bootstrap CIs vs control, all and the Maintain-speed subset.
"""
import glob, json, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
J = "../jepa-nureasoning/"
LABS = ["Safe", "Unsafe", "Suboptimal"]


def ego(p):                       # (N,3) x,y,yaw global -> ego frame of p[0]
    p = np.asarray(p, float); d = p[:, :2] - p[0, :2]; c, s = np.cos(-p[0, 2]), np.sin(-p[0, 2])
    return np.c_[d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c, p[:, 2] - p[0, 2]]


def kin(p):                       # p: (51,3) ego frame incl. t0
    v = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1) / 0.1
    vs = [v[:5].mean(), v[5:10].mean(), v[15:20].mean(), v[25:30].mean(), v[45:50].mean()]
    return vs + [vs[2] - vs[0], vs[4] - vs[0], v.min(), v.min() / (vs[0] + 0.5), p[-1, 0], p[-1, 1], p[-1, 2], p[20, 0]]


FUT = {}
H = json.load(open(J + "data/edge_inputs_validation_dupfix.json"))
FUT["oracle"] = {c: kin(np.r_[[[0, 0, 0]], np.array(r["fut64"])[:50]]) for c, r in H.items() if r["n_valid"] >= 50}
z = np.load(J + "runs/edge_ft3215_dupfix_validation.npz", allow_pickle=True)
FUT["edge3215"] = {str(n): kin(ego(t)) for n, t in zip(z["names"], z["traj"])}
V = json.load(open(J + "runs/handoff/v94_blend_validation_traj.json"))["clips"]
FUT["v94"] = {c: kin(np.r_[[[0, 0, 0]], np.array(v["ego_xyyaw"])][:51]) for c, v in V.items()}

Q = {}
for f in glob.glob("vqa_val/*/*_vqa.json"):
    c = f.split("/")[1]
    for q in json.load(open(f))["questions"]:
        if q["subcategory"] == "action_risk_assessment": q["clip"] = c; Q[q["question_id"]] = q
G = [r for r in map(json.loads, open("artifacts/gbdt_heldout_preds_probs.jsonl")) if r["question_id"] in Q]
G = [r for r in G if all(Q[r["question_id"]]["clip"] in F for F in FUT.values())]
LON = sorted({q["longitudinal"] for q in Q.values()}); LAT = sorted({q["lateral"] for q in Q.values()})
base_x, y, base, clip, grp, ms = [], [], [], [], [], []
for g in G:
    q = Q[g["question_id"]]; gold = q["choices"][q["answer"]]; inv = {v: k for k, v in q["choices"].items()}
    base_x.append([g["probs"].get(inv.get(L, "?"), 0.0) for L in LABS] + [q["longitudinal"] == l for l in LON]
                  + [q["lateral"] == l for l in LAT])
    y.append(LABS.index(gold)); base.append(g["model"] == g["gold"]); clip.append(q["clip"]); grp.append(q["clip"][:19])
    ms.append(q["longitudinal"] == "Maintain speed" and q["lateral"] == "No lateral action")
base_x = np.array(base_x, float); y = np.array(y); base = np.array(base); clip = np.array(clip); grp = np.array(grp)
ms = np.array(ms)
print(f"n {len(y)} (maintain+none {ms.sum()}); GBDT alone: all {base.mean():.3f}, maintain+none {base[ms].mean():.3f}")


def cv(X, seeds=3):
    acc = np.zeros(len(y))
    for sd in range(seeds):
        for tr, te in GroupKFold(5).split(X, y, grp):
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=sd).fit(X[tr], y[tr])
            acc[te] += (m.predict(X[te]) == y[te]) / seeds
    return acc


def ci(d, mk, seed=0):
    cl = clip[mk]; d = d[mk]; uc = np.unique(cl); idx = {c: np.where(cl == c)[0] for c in uc}; rng = np.random.default_rng(seed)
    bs = [np.concatenate([d[idx[c]] for c in rng.choice(uc, len(uc))]).mean() for _ in range(2000)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


ctrl = cv(base_x)
print(f"control (refit, no future): all {ctrl.mean():.3f}, maintain+none {ctrl[ms].mean():.3f}")
for name, F in FUT.items():
    X = np.c_[base_x, np.array([F[c] for c in clip])]; a = cv(X)
    s = " | ".join(f"{nm} {a[mk].mean():.3f} vs ctrl {ci(a - ctrl, mk)[0]:+.3f} [{ci(a - ctrl, mk)[1]:+.3f}, {ci(a - ctrl, mk)[2]:+.3f}]"
                   for nm, mk in (("all", np.ones(len(y), bool)), ("m+n", ms)))
    print(f"{name:9s} {s}")
# how well each predicted future tracks the human one on these clips (5 s speed ratio and min speed)
for name in ("edge3215", "v94"):
    P = np.array([FUT[name][c] for c in clip]); O = np.array([FUT["oracle"][c] for c in clip])
    print(f"{name}: corr with human future, v(4.5-5s) {np.corrcoef(P[:, 4], O[:, 4])[0, 1]:.3f}, min speed "
          f"{np.corrcoef(P[:, 7], O[:, 7])[0, 1]:.3f}, dv(5s) {np.corrcoef(P[:, 6], O[:, 6])[0, 1]:.3f}")
