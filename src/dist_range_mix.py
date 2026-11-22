"""2026-11-13: where does the distance ensemble lose, and does test's mix weight those places more?
Validation: mean (geometric) of the 3 train-only ViT-L nets (ens_l16_s0/1/2, validation never seen in training).
Test: same 3 nets (p19 uses 6 incl. train+val; the 3 train-only ones are the honest proxy for test error rate).
Bins use only INPUTS (predicted distance, object class, camera, box height, option spacing), never test answers.
Expected test errors = sum over bins of (test count x validation error rate in that bin).
"""
import glob, json, math, os, re, numpy as np
from collections import Counter, defaultdict
NETS = ["ens_l16_s0", "ens_l16_s1", "ens_l16_s2"]
P = {s: {} for s in ("validation", "test")}
for s in P:
    ds = [json.load(open(f"runs_dist/{n}/pred_{s}.json")) for n in NETS]
    P[s] = {q: math.exp(np.mean([math.log(d[q]) for d in ds])) for q in ds[0]}
RX = re.compile(r"For the (.+?) in the (.+?) camera with normalized 2D box \[(\d+), (\d+), (\d+), (\d+)\]")


def num(v): return float(v.split()[0])


def feats(q, pred):
    cat, cam, x0, y0, x1, y1 = RX.search(q["question"]).groups(); o = sorted(num(v) for v in q["choices"].values())
    gap = min(math.log(b / a) for a, b in zip(o, o[1:]))
    near = min(q["choices"], key=lambda k: abs(math.log(num(q["choices"][k]) / pred)))
    return dict(cat=cat, cam=cam, h=int(y1) - int(y0), gap=gap, pred=pred, ans=near)


V, T = [], []
for f in glob.glob("vqa_val/*/*_vqa.json"):
    for q in json.load(open(f))["questions"]:
        if q["subcategory"] == "distance_from_2d_bbox" and q["question_id"] in P["validation"]:
            r = feats(q, P["validation"][q["question_id"]]); r["ok"] = r["ans"] == q["answer"]; r["gold"] = num(q["answer_text"]); V.append(r)
for c in os.listdir("cache/test"):
    p = f"cache/test/{c}/reasoning_questions.json"
    if os.path.isfile(p):
        for q in json.load(open(p))["questions"]:
            if q["subcategory"] == "distance_from_2d_bbox": T.append(feats(q, P["test"][q["question_id"]]))
print(f"validation n {len(V)} acc {np.mean([r['ok'] for r in V]):.3f}; test n {len(T)}")


def table(name, key, edges=None):
    def b(r):
        v = r[key]
        if edges is None: return v
        for i, e in enumerate(edges):
            if v < e: return f"<{e}"
        return f">={edges[-1]}"
    vb = defaultdict(list); tb = Counter()
    for r in V: vb[b(r)].append(r["ok"])
    for r in T: tb[b(r)] += 1
    keys = sorted(set(vb) | set(tb), key=lambda k: (edges is None and -tb[k], (edges or [0]) and str(k)))
    if edges is not None: keys = [f"<{e}" for e in edges] + [f">={edges[-1]}"]
    print(f"\n{name}: bin | val n, share, acc | test n, share | expected test errors")
    exp_tot = 0
    for k in keys:
        n, t = len(vb[k]), tb[k]
        if n == 0 and t == 0: continue
        acc = np.mean(vb[k]) if n else float("nan"); e = t * (1 - acc) if n else float("nan"); exp_tot += 0 if n == 0 else e
        print(f"  {str(k):22s} | {n:5d} {n / len(V):.3f} {acc:.3f} | {t:4d} {t / len(T):.3f} | {e:5.1f}")
    print(f"  expected test errors {exp_tot:.1f} of {len(T)} -> acc {1 - exp_tot / len(T):.3f} (val {np.mean([r['ok'] for r in V]):.3f})")


table("predicted distance (m)", "pred", [10, 20, 30, 45, 60, 80])
table("box height (normalized px)", "h", [20, 40, 80, 160])
table("tightest option gap (log ratio)", "gap", [0.1, 0.2, 0.3, 0.5])
table("object class", "cat")
table("camera", "cam")
# calibration of the predicted-distance proxy on validation
g = np.array([r["gold"] for r in V]); p = np.array([r["pred"] for r in V])
for lo, hi in ((0, 20), (20, 45), (45, 80), (80, 1e9)):
    m = (g >= lo) & (g < hi)
    print(f"gold {lo}-{hi} m: n {m.sum()} median |log err| {np.median(np.abs(np.log(p[m] / g[m]))):.3f} median log bias {np.median(np.log(p[m] / g[m])):+.3f}")
