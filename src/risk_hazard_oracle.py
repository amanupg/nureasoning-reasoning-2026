"""2026-11-14 ORACLE check (validation only, nothing for test): does knowing what is around the ego vehicle (lead
vehicle gap / closing speed / time-to-collision, nearest pedestrian or cyclist ahead, occupancy of the adjacent lanes)
improve action_risk beyond the deployed recipe (GBDT x VLM run 1 .5, held-out validation 864)? Hazard features come
from the TRUE key-frame annotations (cache/<split>/<clip>/reasoning/*.json, frame ~100), which test does not have.
If even the oracle adds nothing, a detector-based version for test is not worth building.
HGB fitted on TRAIN action_risk questions: control = ego kinematics (artifacts/clipfeat) + asked action;
hazard = control + hazard features. Validation: standalone accuracy, and fusion with the deployed recipe
p = (1-w) p_deployed + w p_hgb, w chosen by cross-fit over drive logs (20 random halves).
"""
import glob, json, os, random, re, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
PAT = re.compile(r"risk level of: '(.+?)' \(longitudinal\) \+ '(.+?)' \(lateral\)")
SUB = "action_risk_assessment"
VEH = ("vehicle",); VRU = ("human", "pedestrian", "bicycle", "cyclist", "motorcycle")


def qs(split):
    out = []
    for f in glob.glob(f"{'vqa_train' if split == 'train' else 'vqa_val'}/*/*_vqa.json"):
        out += [(f.split("/")[1], q) for q in json.load(open(f))["questions"] if q["subcategory"] == SUB]
    return out


def walk(o):
    if isinstance(o, dict):
        if "position_3d_ego" in o and "track_token" in o: yield o
        for v in o.values(): yield from walk(v)
    elif isinstance(o, list):
        for v in o: yield from walk(v)


def hazard(split, clip, v_ego):
    objs = {}
    for fn in glob.glob(f"cache/{split}/{clip}/reasoning/*.json"):
        R = json.load(open(fn))
        if abs(R.get("frame_index", -99) - 100) > 1: continue
        for o in walk(R): objs[o["track_token"]] = o
    BIG = 99.0
    f = dict(lead_x=BIG, lead_close=0.0, lead_ttc=BIG, vru_x=BIG, vru_absy=BIG, n_near15=0, left_occ=BIG, right_occ=BIG,
             left_rear_close=0.0, right_rear_close=0.0, n_obj=len(objs), have=float(bool(objs)))
    for o in objs.values():
        x, y = o["position_3d_ego"]["x"], o["position_3d_ego"]["y"]; vx = (o.get("velocity_3d_ego") or {}).get("x", 0.0)
        cat = o.get("category", ""); close = v_ego - vx                       # >0: gap shrinking (object ahead)
        if np.hypot(x, y) <= 15: f["n_near15"] += 1
        if cat.startswith(VEH) and 0 < x <= 80 and abs(y) <= 1.8 and x < f["lead_x"]:
            f["lead_x"] = x; f["lead_close"] = close; f["lead_ttc"] = x / close if close > 0.1 else BIG
        if any(k in cat for k in VRU) and -2 < x <= 40 and abs(y) <= 6 and x < f["vru_x"]:
            f["vru_x"] = x; f["vru_absy"] = abs(y)
        for side, sgn in (("left", 1), ("right", -1)):
            if 1.8 < sgn * y <= 5.5 and -15 <= x <= 25:
                f[f"{side}_occ"] = min(f[f"{side}_occ"], abs(x))
                if x < 0: f[f"{side}_rear_close"] = max(f[f"{side}_rear_close"], vx - v_ego)   # >0: catching up from behind
    return f


def main():
    TRQ, VAQ = qs("train"), qs("validation")
    LON = sorted({PAT.search(q["question"]).group(1) for _, q in TRQ}); LAT = sorted({PAT.search(q["question"]).group(2) for _, q in TRQ})
    CF = {s: json.load(open(f"artifacts/clipfeat_{s}.json")) for s in ("train", "validation")}
    EK = ["c_v", "c_ax", "c_ay", "c_omega", "c_dv", "c_vh"]
    H = {}
    def feats(split, c, q, use_h):
        e = CF[split].get(c, {}); ego = [e.get(k, np.nan) for k in EK]
        lon, lat = PAT.search(q["question"]).groups()
        row = ego + [lon == l for l in LON] + [lat == l for l in LAT]
        if use_h:
            if (split, c) not in H: H[(split, c)] = hazard(split, c, e.get("c_v", 0.0))
            row += list(H[(split, c)].values())
        return row
    G = {r["question_id"]: r for r in map(json.loads, open("artifacts/gbdt_heldout_preds_probs.jsonl")) if r["subcategory"] == SUB}
    V = {}
    for f in glob.glob("runs_vlm/qd4b_s3000_val*_s*of4.jsonl"):
        if any(s in f for s in ("gray", "swap", "val150")): continue
        for l in open(f): r = json.loads(l); V[r["question_id"]] = r
    Qv = {q["question_id"]: (c, q) for c, q in VAQ}
    ids = sorted(i for i in G if i in V and i in Qv)
    gold = np.array([G[i]["gold"] for i in ids]); log = np.array([Qv[i][0][:19] for i in ids])
    ms = np.array([PAT.search(Qv[i][1]["question"]).groups() == ("Maintain speed", "No lateral action") for i in ids])
    dep = {i: {k: .5 * G[i]["probs"].get(k, 0) + .5 * V[i]["probs"].get(k, 0) for k in Qv[i][1]["choices"]} for i in ids}
    base = np.array([max(sorted(dep[i]), key=dep[i].get) == g for i, g in zip(ids, gold)])
    print(f"validation held-out n {len(ids)} (maintain+none {ms.sum()}); deployed recipe acc {base.mean():.3f}, m+n {base[ms].mean():.3f}", flush=True)
    GRID = [round(v, 1) for v in np.arange(0, 1.01, 0.1)]
    for name, use_h in (("control (ego + action)", False), ("hazard (oracle annotations)", True)):
        X = np.array([feats("train", c, q, use_h) for c, q in TRQ], float); y = [q["choices"][q["answer"]] for c, q in TRQ]
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=0).fit(X, y)
        P = {}
        for i in ids:
            c, q = Qv[i]; p = dict(zip(m.classes_, m.predict_proba(np.array([feats("validation", c, q, use_h)], float))[0]))
            pr = {L: p.get(t, 0.0) + 1e-6 for L, t in q["choices"].items()}; z = sum(pr.values()); P[i] = {L: v / z for L, v in pr.items()}
        A = {}
        for w in GRID:
            A[w] = np.array([max(sorted(dep[i]), key=lambda k: (1 - w) * dep[i][k] + w * P[i][k]) == g for i, g in zip(ids, gold)])
        logs = sorted(set(log)); rng = random.Random(0); cf, cfm = [], []
        for _ in range(20):
            rng.shuffle(logs); h = set(logs[: len(logs) // 2]); mA = np.array([l in h for l in log])
            for fit, ev in ((mA, ~mA), (~mA, mA)):
                wb = max(GRID, key=lambda w: (A[w][fit].mean(), -w)); cf.append(A[wb][ev].mean() - A[0.0][ev].mean())
                cfm.append(A[wb][ev & ms].mean() - A[0.0][ev & ms].mean())
        print(f"[{name}] train n {len(y)}; standalone acc {A[1.0].mean():.3f} (m+n {A[1.0][ms].mean():.3f}); by w: "
              + " ".join(f"{w}:{A[w].mean():.3f}/{A[w][ms].mean():.3f}" for w in GRID), flush=True)
        print(f"   cross-fit fusion gain all {np.mean(cf):+.4f} (sd {np.std(cf):.4f}, >0 in {np.mean(np.array(cf) > 0):.2f}); "
              f"m+n {np.mean(cfm):+.4f}", flush=True)


if __name__ == "__main__":
    main()
