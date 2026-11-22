"""Distance from box geometry + calibration only (no image), for question boxes whose object is not visible
(SAM 3 check, src/box_visible.py). HGB on log distance, fitted on TRAIN annotation samples, features =
src/box3d_fit.geo (box, size, camera, category, ground-plane lift, size-prior distance, bearing).
Compares with the full-res crop net (runs_dist/net_b16) on validation distance questions, split by visibility.
-> runs_dist/geo_only/pred_{validation,test}.json {question_id: dist}
usage: python src/dist_geo_only.py"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dist_mono import pick
from sklearn.ensemble import HistGradientBoostingRegressor as HGB


def load(stem):
    return [json.loads(l) for l in open(f"artifacts/{stem}.jsonl")], np.load(f"artifacts/{stem}_geo.npy")


def vis(split):
    p = f"runs_vis/{split}.jsonl"
    return {r["question_id"]: (r["best_iou"] >= 0.3 or r["best_cover"] >= 0.5)
            for r in map(json.loads, open(p))} if os.path.isfile(p) else {}


tr, G = load("box3d_ann_train"); k = [i for i, s in enumerate(tr) if (s.get("dist") or 0) > 0.5]
m = HGB(max_iter=800, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=1.0, random_state=0).fit(G[k], np.log([tr[i]["dist"] for i in k]))
print(f"fitted on {len(k)} train samples", flush=True)
os.makedirs("runs_dist/geo_only", exist_ok=True)
for split in ("validation", "test"):
    Q, Gq = load(f"box3d_q_{split}"); idx = [i for i, s in enumerate(Q) if s["sub"] == "distance_from_2d_bbox"]
    pg = np.exp(m.predict(Gq[idx])); out = {Q[i]["qid"]: float(d) for i, d in zip(idx, pg)}
    json.dump(out, open(f"runs_dist/geo_only/pred_{split}.json", "w"))
    if split != "validation": print(split, len(out), "predictions"); continue
    net = json.load(open("runs_dist/net_b16/pred_validation.json")); V = vis("validation")
    rows = []
    for i in idx:
        s = Q[i]; q = s["qid"]
        if q not in net: continue
        t = float(s["choices"][s["answer"]].split()[0])
        rows.append((V.get(q), pick(out[q], s["choices"]) == s["answer"], pick(net[q], s["choices"]) == s["answer"],
                     abs(math.log(out[q] / t)), abs(math.log(net[q] / t))))
    for name, f in (("all", lambda r: True), ("SAM-checked visible", lambda r: r[0] is True), ("SAM-checked NOT visible", lambda r: r[0] is False)):
        r = [x for x in rows if f(x)]
        if r: print(f"  {name:24s} n={len(r):4d} acc geo-only {np.mean([x[1] for x in r]):.3f} net {np.mean([x[2] for x in r]):.3f} | median rel err geo {np.median([x[3] for x in r]):.3f} net {np.median([x[4] for x in r]):.3f}", flush=True)
