"""4-way blends on validation: v7.1, WA-JEPA, fine-tuned Alpamayo (mean of the two out-of-fold
half-models, runs_jepa/a15m_validation.npz) and fine-tuned Drive-JEPA. Same scorer, blend rule
and split-half-by-drive-log report as src/jepa_mix.py. -> artifacts/jepa_mix4_val.npz"""
import os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jepa_mix as J

GRID = [(round(1 - a - b - c, 1), a, b, c) for a in (0.2, 0.3, 0.4) for b in (0.0, 0.1, 0.2, 0.3)
        for c in (0.0, 0.1, 0.2) if 1 - a - b - c >= 0.2 - 1e-9]


def score_clip(args):
    # defined HERE (not jepa_mix.score_clip): spawned workers re-import modules, so the grid must
    # live in this module or they would silently use jepa_mix's 3-way grid.
    name, T = args
    import make_submission as M
    from nureasoning.planning.benchmark import evaluate_clip
    cp = os.path.join("cache/validation", name); out = []
    for w in GRID:
        assert len(w) == len(T)
        r = evaluate_clip(cp, M.CFG, J.blend(T, w))
        out.append(np.nan if r.error else r.s_collision * r.s_driveable * (.3 * r.s_progress + .2 * r.s_comfort + .5 * r.s_human))
    return out
COMP = [("v71", "v71"), ("wa", "wajepa_169"), ("a15m", "a15m"), ("dj", "djft_r40")]


def main():
    Z = {k: np.load(f"runs_jepa/{f}_validation.npz") for k, f in COMP}
    names = sorted(set.intersection(*[set(map(str, z["names"])) for z in Z.values()]))
    idx = {k: {str(n): i for i, n in enumerate(z["names"])} for k, z in Z.items()}
    jobs = [(n, [Z[k]["traj"][idx[k][n]] for k, _ in COMP]) for n in names]
    with ProcessPoolExecutor(int(os.environ.get("NPROC", 8))) as ex:
        N = np.array(list(ex.map(score_clip, jobs, chunksize=4)), float)
    np.savez_compressed("artifacts/jepa_mix4_val.npz", names=np.array(names), grid=np.array(GRID), nps=N)
    report(names, N)


def report(names, N):
    G = [tuple(np.round(w, 1)) for w in GRID]; m = np.nanmean(N, 0)
    ref = G.index((0.7, 0.3, 0.0, 0.0))
    print(f"{len(names)} clips; v8.3 mix (0.7,0.3,0,0) {m[ref]:.4f}; top 8 in-sample:")
    for i in np.argsort(-m)[:8]: print(f"  {G[i]}: {m[i]:.4f}")
    logs = np.array(["_".join(n.split("_")[:2]) for n in names]); ul = np.unique(logs); gains, picks = [], []
    for s in range(20):
        A = np.isin(logs, np.random.RandomState(s).permutation(ul)[: len(ul) // 2])
        for tr, te in ((A, ~A), (~A, A)):
            k = int(np.nanargmax(np.nanmean(N[tr], 0))); picks.append(G[k])
            gains.append(np.nanmean(N[te, k]) - np.nanmean(N[te, ref]))
    g = np.array(gains); from collections import Counter
    print(f"held-out gain over the v8.3 mix (40 halves): {g.mean():+.4f} (sd {g.std():.4f}, >0 in {np.mean(g > 0):.0%})")
    print("picked:", Counter(picks).most_common(5))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        z = np.load("artifacts/jepa_mix4_val.npz"); report(list(z["names"]), z["nps"])
    else:
        main()
