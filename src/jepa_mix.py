"""Blends of planners on validation: per-timestep weighted mean of global x, y and a weighted
circular mean of yaw (the recipe behind v8.3 = 0.7 v7.1 + 0.3 WA-JEPA).

Inputs (runs_jepa/, copied read-only from ../jepa-nureasoning/runs): v71_validation.npz (v7.1 built
from OUT-OF-FOLD chooser picks), wajepa_169_validation.npz, djft_r40_validation.npz.
Scores every blend on a weight grid with the unmodified devkit scorer, then picks weights on one
half of the validation drive logs and scores them on the other half (20 random log splits).
-> artifacts/jepa_mix_val.npz {names, grid, nps (n_clips, n_grid)} and a printed report.
"""
import os, sys, itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
GRID = [(1 - a - b, a, b) for a in np.arange(0, 0.61, 0.1) for b in np.arange(0, 0.61, 0.1) if a + b <= 0.7 + 1e-9]


def blend(T, w):
    xy = sum(wi * t[:, :2] for wi, t in zip(w, T))
    s = sum(wi * np.sin(t[:, 2]) for wi, t in zip(w, T)); c = sum(wi * np.cos(t[:, 2]) for wi, t in zip(w, T))
    return np.column_stack([xy, np.arctan2(s, c)])


def score_clip(args):
    name, T = args
    import make_submission as M
    from nureasoning.planning.benchmark import evaluate_clip
    cp = os.path.join("cache/validation", name); out = []
    for w in GRID:
        r = evaluate_clip(cp, M.CFG, blend(T, w))
        out.append(np.nan if r.error else r.s_collision * r.s_driveable * (.3 * r.s_progress + .2 * r.s_comfort + .5 * r.s_human))
    return out


def main():
    Z = {k: np.load(f"runs_jepa/{f}_validation.npz") for k, f in [("v71", "v71"), ("wa", "wajepa_169"), ("dj", "djft_r40")]}
    names = sorted(set.intersection(*[set(map(str, z["names"])) for z in Z.values()]))
    idx = {k: {str(n): i for i, n in enumerate(z["names"])} for k, z in Z.items()}
    jobs = [(n, [Z[k]["traj"][idx[k][n]] for k in ("v71", "wa", "dj")]) for n in names]
    with ProcessPoolExecutor(int(os.environ.get("NPROC", os.cpu_count()))) as ex:
        N = np.array(list(ex.map(score_clip, jobs, chunksize=4)), float)
    np.savez_compressed("artifacts/jepa_mix_val.npz", names=np.array(names), grid=np.array(GRID), nps=N)
    report(names, N)


def report(names, N):
    g = {tuple(np.round(w, 1)): i for i, w in enumerate(GRID)}
    base, fixed = g[(1.0, 0.0, 0.0)], g[(0.7, 0.3, 0.0)]
    m = np.nanmean(N, 0)
    print(f"{len(names)} validation clips. full-set means (in-sample for the best row):")
    for w in [(1.0, 0.0, 0.0), (0.7, 0.3, 0.0), (0.7, 0.0, 0.3), (0.6, 0.2, 0.2), (0.5, 0.3, 0.2)]:
        print(f"  v71/wa/dj {w}: {m[g[w]]:.4f}")
    best = int(np.nanargmax(m)); print(f"  best on full set {GRID[best]}: {m[best]:.4f}")
    logs = np.array(["_".join(n.split("_")[:2]) for n in names]); ul = np.unique(logs)
    gains_fixed, gains_sel, picks = [], [], []
    for s in range(20):
        half = set(np.random.RandomState(s).permutation(ul)[: len(ul) // 2]); A = np.isin(logs, list(half))
        for tr, te in ((A, ~A), (~A, A)):
            k = int(np.nanargmax(np.nanmean(N[tr], 0))); picks.append(GRID[k])
            gains_sel.append(np.nanmean(N[te, k]) - np.nanmean(N[te, base]))
            gains_fixed.append(np.nanmean(N[te, fixed]) - np.nanmean(N[te, base]))
    gs, gf = np.array(gains_sel), np.array(gains_fixed)
    print(f"held-out halves (40 = 20 log splits x 2): gain over v7.1")
    print(f"  weights picked on the other half: {gs.mean():+.4f} (sd {gs.std():.4f}, >0 in {np.mean(gs > 0):.0%})")
    print(f"  fixed 0.7/0.3:                    {gf.mean():+.4f} (sd {gf.std():.4f}, >0 in {np.mean(gf > 0):.0%})")
    from collections import Counter
    print("  picked weights:", Counter(tuple(np.round(p, 1)) for p in picks).most_common(6))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        z = np.load("artifacts/jepa_mix_val.npz"); report(list(z["names"]), z["nps"])
    else:
        main()
