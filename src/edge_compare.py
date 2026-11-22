"""Edge zero-shot (runs_vlm/edge_val_<tag>_s*.jsonl) vs VLM run 1 (train-only, runs_vlm/qd4b_s3000_val_s*of4.jsonl) and
the GBDT held-out answers, per test type on the same validation questions; paired clip-bootstrap CI for Edge - VLM.
usage: python src/edge_compare.py native6 [all8]"""
import glob, json, sys, numpy as np
from collections import defaultdict
TEST_N = {'image_region_from_3d_object': 400, 'future_motion_label': 400, 'distance_from_2d_bbox': 400, 'action_risk_assessment': 379,
          'camera_view_for_3d_object': 267, 'driving_decision_joint': 190, 'driving_decision_lateral': 171, 'multiview_camera_pair': 163,
          'driving_decision_longitudinal': 139, 'ego_position_from_2d_bbox': 126, 'unsafe_action_identification': 121,
          'object_relative_position': 88, 'future_path_intersection': 61, 'closest_object_type': 42, 'conflict_prediction': 33, 'motion_relation': 20}
V = {}
for f in glob.glob("runs_vlm/qd4b_s3000_val_s*of4.jsonl"):
    for l in open(f): r = json.loads(l); V[r["question_id"]] = r
G = {r["question_id"]: r for r in map(json.loads, open("artifacts/gbdt_heldout_preds_probs.jsonl"))}
for tag in sys.argv[1:]:
    E = {}
    for f in glob.glob(f"runs_vlm/edge_val_{tag}_s*.jsonl"):
        for l in open(f): r = json.loads(l); E[r["question_id"]] = r
    gen_ok = np.mean([r["gen"].strip()[:1] in r["probs"] for r in E.values()])
    print(f"== Edge {tag}: {len(E)} questions, greedy text starts with an offered letter {gen_ok:.3f}, accuracy {np.mean([r['pred'] == r['gold'] for r in E.values()]):.3f}")
    print(f"{'type':32s} {'n':>4s} {'Edge':>6s} | {'nV':>4s} {'Edge':>6s} {'VLM1':>6s} | {'nG':>4s} {'Edge':>6s} {'GBDT':>6s}")
    by = defaultdict(list)
    for q, r in E.items(): by[r["subcategory"]].append(q)
    for t in sorted(by, key=lambda t: -TEST_N.get(t, 0)):
        qs = by[t]; e = np.mean([E[q]["pred"] == E[q]["gold"] for q in qs])
        qv = [q for q in qs if q in V]; qg = [q for q in qs if q in G]
        sv = f"{len(qv):4d} {np.mean([E[q]['pred'] == E[q]['gold'] for q in qv]):6.3f} {np.mean([V[q]['pred'] == V[q]['gold'] for q in qv]):6.3f}" if qv else f"{0:4d} {'':6s} {'':6s}"
        sg = f"{len(qg):4d} {np.mean([E[q]['pred'] == E[q]['gold'] for q in qg]):6.3f} {np.mean([G[q]['model'] == G[q]['gold'] for q in qg]):6.3f}" if qg else ""
        print(f"{t:32s} {len(qs):4d} {e:6.3f} | {sv} | {sg}")
    # test-share weighted, on questions VLM 1 also answered
    qv = [q for q in E if q in V]; w = defaultdict(float)
    for q in qv: w[E[q]["subcategory"]] += 1
    W = np.array([TEST_N.get(E[q]["subcategory"], 0) / w[E[q]["subcategory"]] for q in qv])
    d = np.array([(E[q]["pred"] == E[q]["gold"]) - (V[q]["pred"] == V[q]["gold"]) for q in qv], float)
    cl = np.array([E[q]["clip"] for q in qv]); uc = np.unique(cl); idx = {c: np.where(cl == c)[0] for c in uc}; rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        ii = np.concatenate([idx[c] for c in rng.choice(uc, len(uc))]); bs.append((d[ii] * W[ii]).sum() / W[ii].sum())
    ev = np.array([E[q]["pred"] == E[q]["gold"] for q in qv]); vv = np.array([V[q]["pred"] == V[q]["gold"] for q in qv])
    print(f"test-mix weighted, n {len(qv)} shared with VLM 1: Edge {(ev * W).sum() / W.sum():.3f} VLM1 {(vv * W).sum() / W.sum():.3f} "
          f"diff {(d * W).sum() / W.sum():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
