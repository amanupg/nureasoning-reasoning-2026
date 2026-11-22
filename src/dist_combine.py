"""distance_from_2d_bbox: combine the GBDT's raw distance estimate with the box-conditioned regressor
(log-space weighted mean, weight w on the regressor), nearest choice in log distance. w chosen on one
drive-log half, scored on the other (20 splits x 2), validation held-out questions only.
usage: python src/dist_combine.py [box3d pred json]
"""
import json, math, random, sys
import numpy as np
sys.path.insert(0, "src")
from dist_mono import pick

P = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "artifacts/box3d_pred_validation.json"))
G = [json.loads(l) for l in open("artifacts/gbdt_heldout_preds_est.jsonl")]
Q = {json.loads(l)["qid"]: json.loads(l) for l in open("artifacts/box3d_q_validation.jsonl")}
rows = [g for g in G if g["subcategory"] == "distance_from_2d_bbox" and g["question_id"] in P and g["est"] and g["est"] > 0]
GRID = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
def acc(rs, w):
    return np.mean([pick(math.exp((1 - w) * math.log(r["est"]) + w * math.log(P[r["question_id"]]["dist"])), Q[r["question_id"]]["choices"]) == r["gold"] for r in rs])
print(f"n={len(rows)}  GBDT letters {np.mean([r['model'] == r['gold'] for r in rows]):.3f}  GBDT est (log-nearest) {acc(rows, 0):.3f}  regressor {acc(rows, 1):.3f}")
for w in GRID: print(f"  w={w:.1f} {acc(rows, w):.3f}", end="")
print()
log = lambda r: r["clip"].split("_")[1]; logs = sorted({log(r) for r in rows}); gains = []; ws = []
for seed in range(20):
    A = set(random.Random(seed).sample(logs, len(logs) // 2))
    for h in (True, False):
        fit = [r for r in rows if (log(r) in A) == h]; ev = [r for r in rows if (log(r) in A) != h]
        w = max(GRID, key=lambda x: acc(fit, x)); ws.append(w); gains.append(acc(ev, w) - np.mean([r["model"] == r["gold"] for r in ev]))
print(f"cross-fitted gain vs GBDT letters {np.mean(gains):+.3f} (positive in {np.mean(np.array(gains) > 0):.0%}), chosen w {np.median(ws):.1f}")
for cat in ("car", "traffic_cone", "human"):
    rc = [r for r in rows if Q[r["question_id"]]["cat"] == cat]
    print(f"  {cat:13s} n={len(rc):4d} GBDT {np.mean([r['model'] == r['gold'] for r in rc]):.3f}  w=0.5 {acc(rc, 0.5):.3f}")
