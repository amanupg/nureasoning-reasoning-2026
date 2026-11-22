"""Hybrid scorer: which model should supply each sub-metric head?

Validation is split by DRIVE LOG into two halves; the combination (head sources
+ selection weights) is tuned on one half and reported on the other, then
swapped. Models themselves were fitted only on train.
"""
import glob, itertools, json, numpy as np

ps = [np.load(f) for f in sorted(glob.glob("artifacts/labels148/validation_*.npz"))]
names = np.concatenate([p["names"] for p in ps]).astype(str); Y = np.concatenate([p["labels"] for p in ps])
voc = ps[0]["vocab"]; cur = int(np.where((voc[:, 0] == 0) & (voc[:, 1] == 1.0) & (voc[:, 2] == 0))[0][0])
N = Y[..., 0] * Y[..., 1] * (0.3 * Y[..., 2] + 0.2 * Y[..., 3] + 0.5 * Y[..., 4])
H = ["col", "da", "prog", "comf", "hum"]

def aligned(path, prefix, key_names):
    z = np.load(path); nm = z[key_names].astype(str); pos = {n: i for i, n in enumerate(nm)}
    idx = [pos[n] for n in names]
    return {h: z[f"{prefix}{h}"][idx] for h in H}

G = aligned("artifacts/preds_gbdt148.npz", "eval_", "eval_names")
V = aligned("artifacts/v148_dump/preds_vision.npz", "eval_", "eval_names")
T = aligned("artifacts/v148_dump/preds_novision.npz", "eval_", "eval_names")
SRC = {"gbdt": G, "vision": V, "tf_novision": T,
       "avg(gbdt,vision)": {h: (G[h] + V[h]) / 2 for h in H}}

def score(p, w):
    e = 1e-6
    return w[0] * np.log(p["col"] + e) + w[1] * np.log(p["da"] + e) + np.log(.3 * p["prog"] + .2 * p["comf"] + .5 * p["hum"] + e)

log = np.array(["_".join(n.split("_")[:2]) for n in names]); ul = sorted(set(log))
rng = np.random.RandomState(0); rng.shuffle(ul); A = np.isin(log, ul[:len(ul) // 2]); B = ~A
W = [(a, b) for a in (0, .25, .5, 1, 2, 4, 8) for b in (0, .25, .5, 1, 2, 4, 8)]

def evaluate(cfg):
    """cfg: dict head -> source name. Returns 2-fold held-out NPS."""
    p = {h: SRC[cfg[h]][h] for h in H}
    out = []
    for tune, rep in ((A, B), (B, A)):
        w = max(W, key=lambda w: N[tune][np.arange(tune.sum()), score({h: v[tune] for h, v in p.items()}, w).argmax(1)].mean())
        out.append(N[rep][np.arange(rep.sum()), score({h: v[rep] for h, v in p.items()}, w).argmax(1)].sum())
    return sum(out) / len(N)

print(f"validation {len(N)} clips, {len(ul)} logs; default {N[:, cur].mean():.4f}, oracle {N.max(1).mean():.4f}\n")
rows = []
for gate_src in SRC:
    for rest_src in ("gbdt", "tf_novision"):
        cfg = {"col": gate_src, "da": gate_src, "prog": rest_src, "comf": rest_src, "hum": rest_src}
        rows.append((evaluate(cfg), f"gates from {gate_src:18} | progress/comfort/human from {rest_src}"))
for v, desc in sorted(rows, reverse=True):
    print(f"  {v:.4f}   {desc}")

print("\nper-head swaps (everything else from gbdt):")
for h in H:
    for src in ("vision", "avg(gbdt,vision)"):
        cfg = {k: "gbdt" for k in H}; cfg[h] = src
        print(f"  {evaluate(cfg):.4f}   {h:5} from {src}")
cfg = {k: "gbdt" for k in H}; cfg["prog"] = cfg["hum"] = "avg(gbdt,vision)"
print(f"  {evaluate(cfg):.4f}   prog+hum from avg(gbdt,vision)")

print("\nceiling: replace ONE head with the TRUE label (everything else gbdt):")
TRUE = {h: Y[..., i] for i, h in enumerate(H)}
SRC["TRUE"] = TRUE
for hs in (["col"], ["da"], ["col", "da"], ["prog", "hum"], ["col", "da", "prog", "hum"]):
    cfg = {k: "gbdt" for k in H}
    for h in hs: cfg[h] = "TRUE"
    print(f"  {evaluate(cfg):.4f}   perfect {'+'.join(hs)}")
