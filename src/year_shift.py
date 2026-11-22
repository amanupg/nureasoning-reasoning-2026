"""Does the question->answer relationship differ by recording year? Labelled
train/validation questions only (no test labels exist or are used).
(1) Fit on validation (all 2026) -> accuracy on train questions from 2025 vs 2026 clips.
(2) Fit on train-2026 vs train-2025 (equal size) -> accuracy on held-out train-2025.
Batched prediction per subcategory; at most N questions per (subcategory, year)."""
import collections, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reason_train as RT
import reason_model as RM

N = 3000


def acc(model, rows, maps, prior):
    if not rows: return np.nan
    kind, m, cols = model
    X = np.array([RM.vec(q["_f"], maps) for q in rows], float)[:, cols]
    if kind == "reg":
        d = np.expm1(m.predict(X)); out = []
        for q, di in zip(rows, d):
            cand = {k: RM.num_of(v) for k, v in q["choices"].items()}; cand = {k: v for k, v in cand.items() if v is not None}
            out.append(min(cand, key=lambda k: abs(cand[k] - di)) == q["answer"] if cand else False)
        return np.mean(out)
    P = m.predict_proba(X); cls = m.classes_; out = []
    for q, p in zip(rows, P):
        texts = {k: RM.norm_text(v) for k, v in q["choices"].items()}; pick = None
        for i in np.argsort(-p):
            hit = [k for k, t in texts.items() if t == cls[i]]
            if hit: pick = hit[0]; break
        if pick is None: pick = max(texts, key=lambda k: prior.get(texts[k], 0.0)) if prior else sorted(texts)[0]
        out.append(pick == q["answer"])
    return np.mean(out)


if __name__ != "__main__": raise ImportError("year_shift is a script")
va = RT.prep("vqa_val", "validation"); tr = RT.prep("vqa_train", "train")
maps = RM.build_maps(va + tr)
yr = lambda q: q["_clip"][:4]
log_of = lambda c: "_".join(c.split("_")[:2])
rng = random.Random(0)
by = collections.defaultdict(list)
for q in tr: by[(q["subcategory"], yr(q))].append(q)
mv, pv = RT.fit_all(va, maps)
print(f"{'subcategory':32}{'valfit>26':>10}{'valfit>25':>10}{'t26fit>25':>10}{'t25fit>25':>10}   n25")
for sub in sorted({q["subcategory"] for q in tr}):
    r26 = by[(sub, "2026")]; r25 = by[(sub, "2025")]
    s26 = rng.sample(r26, min(N, len(r26))); s25 = rng.sample(r25, min(N, len(r25)))
    a1 = acc(mv[sub], s26, maps, pv.get(sub, {})) if sub in mv else np.nan
    a2 = acc(mv[sub], s25, maps, pv.get(sub, {})) if sub in mv else np.nan
    # 2025 split by drive log: half to fit a 2025 model, other half to evaluate both
    logs = sorted({log_of(q["_clip"]) for q in r25}); rng.shuffle(logs); h = set(logs[:len(logs) // 2])
    f25 = [q for q in r25 if log_of(q["_clip"]) in h]; e25 = [q for q in r25 if log_of(q["_clip"]) not in h]
    f26 = rng.sample(r26, min(len(f25), len(r26)))
    m25, p25 = RT.fit_all(f25, maps); m26, p26 = RT.fit_all(f26, maps)
    a3 = acc(m26[sub], e25, maps, p26.get(sub, {})) if sub in m26 else np.nan
    a4 = acc(m25[sub], e25, maps, p25.get(sub, {})) if sub in m25 else np.nan
    print(f"{sub:32}{a1:>10.3f}{a2:>10.3f}{a3:>10.3f}{a4:>10.3f}   {len(r25):>5}", flush=True)
