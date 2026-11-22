"""Evaluate / apply DA3 box depth for distance_from_2d_bbox (dist_depth.py output).
Estimate = ego distance of the box-depth percentile point + a physical surface-to-centre offset per
category (half the mean of length and width from TRAIN size medians, artifacts/dist_size_prior_train.json),
optionally rescaled by the image's ground-plane probe ratio. Nearest choice in log distance.
usage: python src/dist_eval.py <prefix> [--gbdt artifacts/gbdt_heldout_preds.jsonl]
"""
import argparse, glob, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dist_mono import pick
PRIOR = json.load(open("artifacts/dist_size_prior_train.json"))
CAT = {"car": "vehicle.car", "human": "human", "traffic_cone": "construction.traffic_cone"}


def offset(cat):
    p = PRIOR.get(CAT.get(cat, cat))
    return (p[0] + p[1]) / 4 if p else 0.0


def estimate(r, stat="f0.6_p25", probe=None, off=True):
    d = r["depth"][stat][1]
    if probe:
        s = r["probes"][probe][0]
        if np.isfinite(s) and r["probes"][probe][1] >= 10 and 0.5 < s < 2: d = d / s
    return d + (offset(r["cat"]) if off else 0.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("prefix"); ap.add_argument("--gbdt", default="artifacts/gbdt_heldout_preds.jsonl")
    a = ap.parse_args()
    R = {}
    for f in glob.glob(f"{a.prefix}_s*of*.json"): R.update(json.load(open(f)))
    R = {q: r for q, r in R.items() if not r.get("missing")}
    G = {json.loads(l)["question_id"]: json.loads(l) for l in open(a.gbdt)} if os.path.isfile(a.gbdt) else {}
    inG = [q for q in R if q in G]
    print(f"{len(R)} questions; GBDT held-out on {len(inG)}: {np.mean([G[q]['model'] == G[q]['gold'] for q in inG]):.3f}")
    probes = [None] + sorted(next(iter(R.values()))["probes"])
    for stat in sorted(next(iter(R.values()))["depth"]):
        for probe in probes:
            for off in (False, True):
                acc = np.mean([pick(estimate(r, stat, probe, off), r["choices"]) == r["answer"] for r in R.values()])
                accg = np.mean([pick(estimate(R[q], stat, probe, off), R[q]["choices"]) == G[q]["gold"] for q in inG]) if inG else float("nan")
                tv = [float(R[q]["choices"][R[q]["answer"]].split()[0]) for q in R]
                rel = np.median([abs(math.log(max(estimate(r, stat, probe, off), 0.1) / max(t, 0.1))) for r, t in zip(R.values(), tv)])
                print(f"{stat:9s} probe={str(probe):10s} off={int(off)}: acc {acc:.3f} (GBDT set {accg:.3f}) med|log err| {rel:.3f}")


if __name__ == "__main__":
    main()
