"""2026-11-13: DINOv3 ViT-H+ distance nets (ens_h16_s0/1/2) vs ViT-L (ens_l16_s0/1/2), all train-only, paired on
validation distance questions; also reweighted to test's predicted-range mix (inputs only, no test answers)."""
import glob, json, math, os, re, numpy as np
RX = re.compile(r"For the (.+?) in the (.+?) camera")
def num(v): return float(v.split()[0])
def load(nets, s):
    ds = [json.load(open(f"runs_dist/{n}/pred_{s}.json")) for n in nets]
    return {q: math.exp(np.mean([math.log(d[q]) for d in ds])) for q in ds[0]}
L = [f"ens_l16_s{i}" for i in range(3)]; H = [f"ens_h16_s{i}" for i in range(3)]
E = {"L": L, "H": H, "LH": L + H}
PV = {k: load(v, "validation") for k, v in E.items()}; PT = {k: load(v, "test") for k, v in E.items()}
def pick(q, p): return min(q["choices"], key=lambda k: abs(math.log(num(q["choices"][k]) / p)))
V = []
for f in glob.glob("vqa_val/*/*_vqa.json"):
    for q in json.load(open(f))["questions"]:
        if q["subcategory"] == "distance_from_2d_bbox" and q["question_id"] in PV["L"]:
            V.append({k: pick(q, PV[k][q["question_id"]]) == q["answer"] for k in E} | {"r": PV["L"][q["question_id"]]})
T = []
for c in os.listdir("cache/test"):
    p = f"cache/test/{c}/reasoning_questions.json"
    if os.path.isfile(p):
        for q in json.load(open(p))["questions"]:
            if q["subcategory"] == "distance_from_2d_bbox":
                T.append({"r": PT["L"][q["question_id"]], **{k: pick(q, PT[k][q["question_id"]]) for k in E}})
edges = [0, 10, 20, 30, 45, 1e9]
b = lambda r: next(i for i in range(5) if edges[i] <= r < edges[i + 1])
print(f"validation n {len(V)}; test n {len(T)}")
for k in E: print(f"  {k:3s} acc {np.mean([v[k] for v in V]):.4f}")
tw = np.bincount([b(t["r"]) for t in T], minlength=5) / len(T)
print("bin   n_val   L      H      LH    test_share")
for i in range(5):
    s = [v for v in V if b(v["r"]) == i]
    print(f"{edges[i]:>3}-{edges[i+1] if edges[i+1] < 1e8 else 'inf':<4} {len(s):5d} " + " ".join(f"{np.mean([v[k] for v in s]):.3f}" for k in E) + f"  {tw[i]:.3f}")
rng = np.random.default_rng(0); idx = [np.array([j for j, v in enumerate(V) if b(v["r"]) == i]) for i in range(5)]
A = {k: np.array([v[k] for v in V], float) for k in E}
for k in ("H", "LH"):
    d = A[k] - A["L"]; tdw = sum(tw[i] * d[idx[i]].mean() for i in range(5) if len(idx[i]))
    bs = [sum(tw[i] * d[rng.choice(idx[i], len(idx[i]))].mean() for i in range(5) if len(idx[i])) for _ in range(4000)]
    print(f"{k} - L: plain {d.mean():+.4f} (wins {int((d>0).sum())} losses {int((d<0).sum())}); test-mix weighted {tdw:+.4f} CI [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}] -> x{len(T)} = {tdw*len(T):+.1f} answers")
for k in ("H", "LH"): print(f"test answers changed vs L: {k} {sum(t[k] != t['L'] for t in T)}")
