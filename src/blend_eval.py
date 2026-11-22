"""Score fixed blends of GLOBAL trajectory files on validation with the devkit scorer (same blend rule
as src/jepa_mix.py) and compare two of them paired, overall and on held-out-style drive-log halves.
usage: python src/blend_eval.py  (components and blends set below; -> artifacts/blend_eval_val.npz)"""
import os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jepa_mix as J

COMP = [("v71", "runs_jepa/v71_validation.npz"), ("v71B", "runs_handoff/v71B_validation.npz"),
        ("wa", "runs_jepa/wajepa_169_validation.npz"), ("dj3", "runs_jepa/djft_r2avg_validation.npz"),
        ("b_raw", "runs_handoff/a15s1lora_raw_validation.npz")]
BLENDS = {"v71": (1, 0, 0, 0, 0), "v71B": (0, 1, 0, 0, 0), "v9.2": (.6, 0, .268, .132, 0),
          "v9.2-B": (0, .6, .268, .132, 0), "v9.2-B+raw": (0, .5, .268, .132, .1)}


def score(args):
    name, T = args
    import make_submission as M
    from nureasoning.planning.benchmark import evaluate_clip
    out = []
    for w in BLENDS.values():
        r = evaluate_clip(os.path.join("cache/validation", name), M.CFG, J.blend(T, w))
        out.append(np.nan if r.error else r.s_collision * r.s_driveable * (.3 * r.s_progress + .2 * r.s_comfort + .5 * r.s_human))
    return out


def main():
    Z = [np.load(p) for _, p in COMP]
    names = sorted(set.intersection(*[set(map(str, z["names"])) for z in Z]))
    idx = [{str(n): i for i, n in enumerate(z["names"])} for z in Z]
    jobs = [(n, [z["traj"][ix[n]] for z, ix in zip(Z, idx)]) for n in names]
    with ProcessPoolExecutor(int(os.environ.get("NPROC", 8))) as ex:
        N = np.array(list(ex.map(score, jobs, chunksize=4)), float)
    np.savez_compressed("artifacts/blend_eval_val.npz", names=np.array(names), blends=np.array(list(BLENDS)), nps=N)
    keys = list(BLENDS); m = np.nanmean(N, 0)
    for k, v in zip(keys, m): print(f"{k:12s} {v:.4f}")
    a, b = keys.index("v9.2"), keys.index("v9.2-B")
    logs = np.array(["_".join(n.split("_")[:2]) for n in names]); ul = np.unique(logs); d = []
    for s in range(20):
        A = np.isin(logs, np.random.RandomState(s).permutation(ul)[: len(ul) // 2])
        for h in (A, ~A): d.append(np.nanmean(N[h, b]) - np.nanmean(N[h, a]))
    d = np.array(d); print(f"v9.2-B minus v9.2 on drive-log halves: {d.mean():+.4f} (sd {d.std():.4f}, >0 in {np.mean(d > 0):.0%})")


if __name__ == "__main__":
    main()
