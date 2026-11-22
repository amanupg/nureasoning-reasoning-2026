"""Compare a nuVLA benchmark report with our GBDT pick on the SAME validation clips."""
import json, sys, glob, numpy as np
rep = json.load(open(sys.argv[1]))
clips = rep.get("clips") or rep.get("per_clip") or rep
rows = clips if isinstance(clips, list) else list(clips.values())
get = lambda r, *ks: next((r[k] for k in ks if k in r), None)
vla = {get(r, "clip_name", "clip"): get(r, "planning_score", "nps") for r in rows if isinstance(r, dict)}
SEL = "artifacts/versioned/selections_test_novision148_20260926T050149Z.json"
w = json.load(open(SEL))["w"]; e = 1e-6
P = np.load("artifacts/preds_gbdt148.npz")
S = w[0]*np.log(P["eval_col"]+e) + w[1]*np.log(P["eval_da"]+e) + np.log(.3*P["eval_prog"]+.2*P["eval_comf"]+.5*P["eval_hum"]+e)
ps = [np.load(f) for f in sorted(glob.glob("artifacts/labels148/validation_[0-9]*.npz"))]
names = np.concatenate([p["names"] for p in ps]); lab = np.concatenate([p["labels"] for p in ps])
li = {n: i for i, n in enumerate(names)}
nps = lambda L: L[..., 0]*L[..., 1]*(.3*L[..., 2]+.2*L[..., 3]+.5*L[..., 4])
common = [n for n in P["eval_names"] if n in vla and n in li and vla[n] is not None]
g = np.array([nps(lab[li[n]])[S[i].argmax()] for i, n in enumerate(P["eval_names"]) if n in set(common)])
v = np.array([vla[n] for n in P["eval_names"] if n in set(common)])
print(f"{len(common)} common clips:  nuVLA {v.mean():.4f}   GBDT pick {g.mean():.4f}   "
      f"per-clip max(both) {np.maximum(v, g).mean():.4f}")
