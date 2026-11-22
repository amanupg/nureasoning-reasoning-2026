"""2026-11-15: planner-trajectory probabilities from K16 (e2a K=16 mean, jepa runs/edge_e2a_k16_*) vs the deployed K6
v9.9 ones, inside the deployed recipes p = .5 p_fused + .5 p_traj (decisions, unsafe). Held-out validation
(artifacts/for_planning/validation_reasoning_probs.jsonl p_fused), paired, clip bootstrap.
usage: python src/k16_traj_cmp.py"""
import json, numpy as np
R = [json.loads(l) for l in open("artifacts/for_planning/validation_reasoning_probs.jsonl")]
K6 = {k: json.load(open(f"artifacts/{k}_traj_probs_validation.json")) for k in ("dec", "unsafe")}
K16 = {k: json.load(open(f"artifacts/{k}_traj_probs_k16_validation.json")) for k in ("dec", "unsafe")}
for sub in ("driving_decision_joint", "driving_decision_lateral", "driving_decision_longitudinal", "unsafe_action_identification"):
    tk = "unsafe" if sub.startswith("unsafe") else "dec"
    rows = [r for r in R if r["subcategory"] == sub and r["question_id"] in K6[tk] and r["question_id"] in K16[tk]]
    def acc(T, w=.5):
        return np.array([max(sorted(r["p_fused"]), key=lambda k: (1 - w) * r["p_fused"][k] + w * T[r["question_id"]][k]) == r["gold"] for r in rows], float)
    a6, a16 = acc(K6[tk]), acc(K16[tk]); d = a16 - a6; clip = np.array([r["clip"] for r in rows])
    uc = sorted(set(clip)); idx = {c: np.where(clip == c)[0] for c in uc}; br = np.random.default_rng(0)
    bs = [np.concatenate([d[idx[c]] for c in br.choice(uc, len(uc))]).sum() for _ in range(2000)]
    t6 = acc(K6[tk], 1).mean(); t16 = acc(K16[tk], 1).mean()
    print(f"{sub[:30]:30s} n {len(rows)} traj-only K6 {t6:.3f} K16 {t16:.3f} | fused w.5 K6 {a6.mean():.3f} K16 {a16.mean():.3f} "
          f"net {int(d.sum()):+d} [{np.percentile(bs, 2.5):+.0f}, {np.percentile(bs, 97.5):+.0f}]")
