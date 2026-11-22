"""Per-split standalone quality of path sources (raw / own / oracle) on route clips, to check
that train (out-of-fold) and validation predictions of a source look alike."""
import glob, sys, numpy as np
nps = lambda y: y[..., 0] * y[..., 1] * (.3 * y[..., 2] + .2 * y[..., 3] + .5 * y[..., 4])
def cat(d, s):
    P = [np.load(f) for f in sorted(glob.glob(f"artifacts/{d}/{s}_[0-9]*.npz"))]
    return {k: np.concatenate([p[k] for p in P]) for k in ("names", "labels", "a15feat") if k in P[0].files}
for s, dirs in [("train", sys.argv[1].split(",")), ("validation", sys.argv[2].split(","))]:
    b = cat("labels148", s); B = set(b["names"])
    for d in dirs:
        x = cat(d, s); A = nps(x["labels"]); m = np.isin(x["names"], list(B))
        f = x["a15feat"][m]
        print(f"{s:10s} {d:18s} n={m.sum():5d} K={A.shape[1]} raw {A[m,-1].mean():.4f} own {A[m,26].mean():.4f} "
              f"oracle {A[m].max(1).mean():.4f} | feat mean {np.nanmean(f,0).round(2)} nan {np.isnan(f).mean():.3f}", flush=True)
