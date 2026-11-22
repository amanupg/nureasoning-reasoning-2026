"""Per-type accuracy of VLM letter predictions (src/vlm_infer.py, validation) vs the GBDT held-out log
(artifacts/gbdt_heldout_preds.jsonl) on the same questions, test-share weighted.
usage: python src/vlm_eval.py <vlm val prefix>
"""
import glob, json, sys
from collections import defaultdict
sys.path.insert(0, "src")
from vlm_data import TEST_SHARE

V = {}
for f in glob.glob(f"{sys.argv[1]}_s*of*.jsonl"):
    for l in open(f): r = json.loads(l); V[r["question_id"]] = r
G = {json.loads(l)["question_id"]: json.loads(l) for l in open("artifacts/gbdt_heldout_preds.jsonl")}
by = defaultdict(lambda: [0, 0, 0, 0])      # n_vlm, vlm correct, n_both, gbdt correct on both / vlm correct on both
both = defaultdict(lambda: [0, 0, 0])
for q, r in V.items():
    s = r["subcategory"]; by[s][0] += 1; by[s][1] += r["pred"] == r["gold"]
    if q in G: both[s][0] += 1; both[s][1] += r["pred"] == r["gold"]; both[s][2] += G[q]["model"] == G[q]["gold"]
print(f"{'type':32s} share   VLM (n)        | same questions: VLM   GBDT  (n)")
wv = wg = wb = 0
for s in sorted(TEST_SHARE, key=lambda s: -TEST_SHARE[s]):
    n, c = by[s][0], by[s][1]; nb, vb, gb = both[s]
    av = c / n if n else float("nan"); v2 = vb / nb if nb else float("nan"); g2 = gb / nb if nb else float("nan")
    print(f"{s:32s} {100 * TEST_SHARE[s]:4.1f}  {av:.3f} ({n:4d})  | {v2:.3f}  {g2:.3f}  ({nb})")
    if nb: wv += TEST_SHARE[s] * v2; wg += TEST_SHARE[s] * g2; wb += TEST_SHARE[s] * max(v2, g2)
print(f"test-share weighted on shared questions: VLM {wv:.3f}  GBDT {wg:.3f}  best-per-type {wb:.3f}")
