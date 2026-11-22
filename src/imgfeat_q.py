"""Per-question image features from the cached DINOv3 grids (feat/<model>/<split>/<clip>.npy,
[frame (100, 90), cam, CLS + 6x8 grid, 768]) for the reasoning models. CPU only.

Pre-registered (chosen before any result, from the existing per-subcategory accuracies):
- OBJ_SUBS (box questions below 0.95): grid cells under the stated 2D box in the stated
  camera at the key frame, and their change from 1 s earlier.
- SCENE_SUBS (scene-level questions): CLS of front / front_left / front_right at the key
  frame, and the front CLS change from 1 s earlier.
Each block is reduced by PCA fitted on TRAIN questions only; the PCA is saved with the output.
Output: artifacts/imgfeat_<split>.npz {clip, qid, X (float16), cols} + artifacts/imgfeat_pca.npz
usage: python src/imgfeat_q.py [train validation test]
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qfeat import load, feats

FEAT = "feat/dinov3-vitb16-pretrain-lvd1689m"
CAMS = ["back", "back_left", "back_right", "front", "front_left", "front_right", "left", "right"]
OBJ_SUBS = {"distance_from_2d_bbox", "future_motion_label", "closest_object_type"}
SCENE_SUBS = {"action_risk_assessment", "unsafe_action_identification", "driving_decision_joint",
              "driving_decision_lateral", "driving_decision_longitudinal"}
NCOMP = {"obj": 24, "objd": 12, "scn": 24, "scnd": 8}
ROOTS = {"train": "vqa_train", "validation": "vqa_val"}


def test_questions():
    out = []
    for d in sorted(os.listdir("cache/test")):
        p = os.path.join("cache/test", d, "reasoning_questions.json")
        if not os.path.isfile(p): continue
        for q in (json.load(open(p)).get("questions") or []):
            if q.get("question_type", "choice") == "choice":
                q["_clip"] = d; q["_f"] = feats(q); out.append(q)
    return out


def raw(q, g):
    """Raw (unreduced) vectors for one question, or {} if not applicable."""
    s, f = q.get("subcategory"), q["_f"]
    if s in OBJ_SUBS and "bx" in f and f.get("cam") in CAMS:
        c = CAMS.index(f["cam"])
        x0, x1 = (f["bx"] - f["bw"] / 2) / 1000 * 8, (f["bx"] + f["bw"] / 2) / 1000 * 8
        y0, y1 = (f["by"] - f["bh"] / 2) / 1000 * 6, (f["by"] + f["bh"] / 2) / 1000 * 6
        cols = range(max(0, int(np.floor(x0))), min(8, int(np.floor(x1)) + 1))
        rows = range(max(0, int(np.floor(y0))), min(6, int(np.floor(y1)) + 1))
        idx = [1 + r * 8 + cc for r in rows for cc in cols] or [1 + min(5, int(f["by"] / 1000 * 6)) * 8 + min(7, int(f["bx"] / 1000 * 8))]
        a, b = g[0, c, idx].astype(np.float32).mean(0), g[1, c, idx].astype(np.float32).mean(0)
        return {"obj": a, "objd": a - b}
    if s in SCENE_SUBS:
        fr, fl, rr = CAMS.index("front"), CAMS.index("front_left"), CAMS.index("front_right")
        v = np.concatenate([g[0, fr, 0], g[0, fl, 0], g[0, rr, 0]]).astype(np.float32)
        return {"scn": v, "scnd": (g[0, fr, 0] - g[1, fr, 0]).astype(np.float32)}
    return {}


def collect(split):
    qs = test_questions() if split == "test" else load(ROOTS[split])
    R, cache = [], {}
    for q in qs:
        c = q["_clip"]
        if c not in cache:
            p = os.path.join(FEAT, split, c + ".npy")
            cache = {c: np.load(p) if os.path.isfile(p) else None}      # questions are grouped by clip
        g = cache[c]
        if g is None: continue
        r = raw(q, g)
        if r: R.append((c, q.get("question_id"), r))
    return R


def main():
    splits = sys.argv[1:] or ["train", "validation", "test"]
    t0 = time.time()
    tr = collect("train")
    print(f"train: {len(tr)} questions with image features ({time.time()-t0:.0f}s)", flush=True)
    pca = {}
    rng = np.random.RandomState(0)
    for k, n in NCOMP.items():
        M = np.stack([r[k] for _, _, r in tr if k in r])
        M = M[rng.choice(len(M), min(len(M), 60000), replace=False)]
        mu = M.mean(0); U, S, Vt = np.linalg.svd(M - mu, full_matrices=False)
        pca[k] = (mu, Vt[:n]); print(f"  PCA {k}: {len(M)} rows, {n} comps, var kept {(S[:n]**2).sum()/(S**2).sum():.3f}")
    os.makedirs("artifacts", exist_ok=True)
    np.savez("artifacts/imgfeat_pca.npz", **{f"{k}_mu": v[0] for k, v in pca.items()}, **{f"{k}_V": v[1] for k, v in pca.items()})
    cols = [f"img_{k}{i}" for k, n in NCOMP.items() for i in range(n)]
    for split in splits:
        R = tr if split == "train" else collect(split)
        X = np.full((len(R), len(cols)), np.nan, np.float32); o = 0
        for k, n in NCOMP.items():
            mu, V = pca[k]
            for i, (_, _, r) in enumerate(R):
                if k in r: X[i, o:o + n] = (r[k] - mu) @ V.T
            o += n
        np.savez_compressed(f"artifacts/imgfeat_{split}.npz", clip=np.array([r[0] for r in R]),
                            qid=np.array([str(r[1]) for r in R]), X=X.astype(np.float16), cols=np.array(cols))
        print(f"{split}: {len(R)} questions -> artifacts/imgfeat_{split}.npz ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
