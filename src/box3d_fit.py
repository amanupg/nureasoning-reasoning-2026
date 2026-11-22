"""Box-conditioned 3D regressor: distance and closing speed of the object in a stated 2D box.
Features: DINOv3 crop embeddings (key frame + older frame, PCA 128 fitted on train) + calibration
geometry (box, camera, category, ground-plane lift of the box bottom, size-prior distance, bearing).
Targets: log distance, closing speed. HistGradientBoosting, fitted on TRAIN annotation samples.
Evaluation on validation QUESTIONS: distance (nearest choice in log) and future motion (0.5 m/s rule),
vs the GBDT held-out log and the SAM tracker on the same questions.
-> artifacts/box3d_pred_{validation,test}.json {question_id: {dist, closing}}
usage: python src/box3d_fit.py [--with-val-ann]   (adds validation annotations to training; then no val eval)
"""
import argparse, glob, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camgeo
from dist_mono import ground, by_height, pick
from fm_rule import rule

PRIOR = json.load(open("artifacts/dist_size_prior_train.json"))
CATMAP = {"car": "vehicle.car", "human": "human", "traffic_cone": "construction.traffic_cone", "bicycle": "vehicle.personal_mobility.bicycle"}
CATS = ["car", "human", "traffic_cone", "bicycle"]; CAMS = list(camgeo.NAME2KEY)
_RIG = {}


def rig(split, clip):
    if (split, clip) not in _RIG:
        _RIG[(split, clip)] = camgeo.Rig(json.load(open(f"cache/{split}/{clip}/metadata.json")))
    return _RIG[(split, clip)]


def geo(s):
    b = s["box"]; r = rig(s["split"], s["clip"]); cam = s["cam"]
    w, h = max(b[2] - b[0], 1e-3), max(b[3] - b[1], 1e-3)
    f = [b[0], b[1], b[2], b[3], math.log(w), math.log(h), (b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
    f += [float(cam == c) for c in CAMS] + [float(s["cat"] == c) for c in CATS]
    if cam in r.cams:
        g = ground(r, cam, (b[0] + b[2]) / 2, b[3]); H = (PRIOR.get(CATMAP.get(s["cat"], ""), [0, 0, 1.5]))[2]
        hz = by_height(r, cam, b, H); d, o = r.ray(cam, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        f += [math.log(g) if g and g > 0 and np.isfinite(g) else np.nan, math.log(hz) if hz and hz > 0 and np.isfinite(hz) else np.nan,
              math.atan2(d[1], d[0]), d[2]]
    else:
        f += [np.nan] * 4
    return f


def load(stem):
    S = [json.loads(l) for l in open(f"artifacts/{stem}.jsonl")]; Z = np.load(f"artifacts/{stem}.npz")
    return S, Z["emb"].astype(np.float32), Z["ok"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--with-val-ann", action="store_true"); a = ap.parse_args()
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    tr, Etr, oktr = load("box3d_ann_train")
    if a.with_val_ann:
        va_, Eva_, okva_ = load("box3d_ann_validation"); tr += va_; Etr = np.vstack([Etr, Eva_]); oktr = np.concatenate([oktr, okva_])
    keep = [i for i in range(len(tr)) if oktr[i] and tr[i]["dist"] > 0.5]
    pca = PCA(128, random_state=0).fit(Etr[keep][::3])
    X = np.hstack([pca.transform(Etr[keep]), np.array([geo(tr[i]) for i in keep])])
    yd = np.log(np.array([tr[i]["dist"] for i in keep]))
    ic = [j for j, i in enumerate(keep) if tr[i]["closing"] is not None]
    yc = np.clip(np.array([tr[keep[j]]["closing"] for j in ic]), -20, 20)
    md = HGB(max_iter=600, learning_rate=0.06, max_leaf_nodes=63, l2_regularization=1.0, random_state=0).fit(X, yd)
    mc = HGB(max_iter=600, learning_rate=0.06, max_leaf_nodes=63, l2_regularization=1.0, random_state=0).fit(X[ic], yc)
    print(f"fitted on {len(keep)} train samples ({len(ic)} with closing speed)", flush=True)
    G = {json.loads(l)["question_id"]: json.loads(l) for l in open("artifacts/gbdt_heldout_preds.jsonl")}
    trk = {}
    for f in glob.glob("runs_fm/sam21_full_s*of4.json"): trk.update(json.load(open(f)))
    for split in ("validation", "test"):
        Q, E, ok = load(f"box3d_q_{split}")
        idx = [i for i in range(len(Q)) if ok[i]]
        Xq = np.hstack([pca.transform(E[idx]), np.array([geo(Q[i]) for i in idx])])
        pd_, pc_ = np.exp(md.predict(Xq)), mc.predict(Xq)
        out = {Q[i]["qid"]: {"dist": float(d), "closing": float(c)} for i, d, c in zip(idx, pd_, pc_)}
        json.dump(out, open(f"artifacts/box3d_pred_{split}{'_tv' if a.with_val_ann else ''}.json", "w"))
        if split != "validation" or a.with_val_ann: print(split, len(out), "predictions"); continue
        for sub in ("distance_from_2d_bbox", "future_motion_label"):
            qs = [Q[i] for i in idx if Q[i]["sub"] == sub]
            if sub == "distance_from_2d_bbox":
                acc = lambda q: pick(out[q["qid"]]["dist"], q["choices"]) == q["answer"]
                med = np.median([abs(math.log(out[q["qid"]]["dist"] / float(q["choices"][q["answer"]].split()[0]))) for q in qs])
            else:
                acc = lambda q: rule(out[q["qid"]]["closing"]) == q["choices"][q["answer"]]
                med = float("nan")
            both = [q for q in qs if q["qid"] in G]
            line = (f"{sub:24s} n={len(qs)} regressor {np.mean([acc(q) for q in qs]):.3f} (median |log err| {med:.3f}) | "
                    f"GBDT-subset n={len(both)}: regressor {np.mean([acc(q) for q in both]):.3f} GBDT {np.mean([G[q['qid']]['model'] == G[q['qid']]['gold'] for q in both]):.3f}")
            if sub == "future_motion_label":
                tq = [q for q in qs if q["qid"] in trk]
                from fm_rule import closing as tclose
                tv = []
                for q in tq:
                    r = trk[q["qid"]]; e = tclose(r, rig("validation", q["clip"]), "h", edge=3, min_frames=2)[0] if not r.get("missing") else float("nan")
                    tv.append((q, e))
                tv = [(q, e) for q, e in tv if np.isfinite(e)]
                line += f" | tracker-usable n={len(tv)}: regressor {np.mean([acc(q) for q, _ in tv]):.3f} tracker {np.mean([rule(e) == q['choices'][q['answer']] for q, e in tv]):.3f}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
