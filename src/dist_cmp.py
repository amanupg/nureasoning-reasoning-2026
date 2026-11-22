"""2026-11-14: paired validation comparison of distance-net recipes (train-only nets), plain and reweighted to test's
predicted range mix (inputs only). A recipe = '+'-joined net dirs, optional ':flip' suffix = geometric mean with the
horizontal-flip predictions (src/dist_tta.py). Also counts test answers changed vs the first recipe.
usage: python src/dist_cmp.py ens_h16_s0+ens_h16_s1+ens_h16_s2 ens_h16_s0+ens_h16_s1+ens_h16_s2:flip ...
"""
import glob, json, math, os, sys, numpy as np
def num(v): return float(v.split()[0])
def preds(recipe, split):
    nets, _, mode = recipe.partition(":"); fs = []
    for n in nets.split("+"):
        fs.append(f"runs_dist/{n}/pred_{split}.json")
        if mode == "flip": fs.append(f"runs_dist/{n}/pred_{split}_flip.json")
        if mode == "fliponly": fs[-1] = f"runs_dist/{n}/pred_{split}_flip.json"
    ds = [json.load(open(f)) for f in fs]
    return {q: math.exp(np.mean([math.log(d[q]) for d in ds])) for q in ds[0]}
R = sys.argv[1:]; PV = {r: preds(r, "validation") for r in R}; PT = {r: preds(r, "test") for r in R}
pick = lambda q, p: min(sorted(q["choices"]), key=lambda k: abs(math.log(num(q["choices"][k]) / p)))
V = []
for f in glob.glob("vqa_val/*/*_vqa.json"):
    for q in json.load(open(f))["questions"]:
        if q["subcategory"] == "distance_from_2d_bbox" and all(q["question_id"] in PV[r] for r in R):
            V.append({r: pick(q, PV[r][q["question_id"]]) == q["answer"] for r in R} | {"rng": PV[R[0]][q["question_id"]]})
T = []
for c in os.listdir("cache/test"):
    p = f"cache/test/{c}/reasoning_questions.json"
    if os.path.isfile(p):
        for q in json.load(open(p))["questions"]:
            if q["subcategory"] == "distance_from_2d_bbox": T.append({"rng": PT[R[0]][q["question_id"]], **{r: pick(q, PT[r][q["question_id"]]) for r in R}})
edges = [0, 10, 20, 30, 45, 1e9]; b = lambda x: next(i for i in range(5) if edges[i] <= x < edges[i + 1])
tw = np.bincount([b(t["rng"]) for t in T], minlength=5) / len(T); idx = [np.array([j for j, v in enumerate(V) if b(v["rng"]) == i]) for i in range(5)]
A = {r: np.array([v[r] for v in V], float) for r in R}
print(f"validation n {len(V)}; test n {len(T)}")
for r in R: print(f"  {r:60s} acc {A[r].mean():.4f} | by range " + " ".join(f"{A[r][ix].mean():.3f}" for ix in idx))
rng = np.random.default_rng(0)
for r in R[1:]:
    d = A[r] - A[R[0]]; tdw = sum(tw[i] * d[idx[i]].mean() for i in range(5) if len(idx[i]))
    bs = [sum(tw[i] * d[rng.choice(idx[i], len(idx[i]))].mean() for i in range(5) if len(idx[i])) for _ in range(4000)]
    print(f"{r} - {R[0]}: plain {d.mean():+.4f} (wins {int((d > 0).sum())} losses {int((d < 0).sum())}); test-mix {tdw:+.4f} "
          f"CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] -> x400 = {tdw * 400:+.1f}; test changed {sum(t[r] != t[R[0]] for t in T)}")
