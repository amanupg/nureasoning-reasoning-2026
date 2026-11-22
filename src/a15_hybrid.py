"""Alpamayo path x our speed choice. The GBDT pick (v4.1 model, never trained on
validation) supplies HOW FAR (arc length of the selected candidate); Alpamayo's
mean path supplies WHERE. Also: oracle over {148 candidates} U {a15 path x each
distinct vocabulary arc length}, the headroom if the scorer learned to use it."""
import glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a15_calib import prep, score

SEL = "artifacts/versioned/selections_test_novision148_20260926T050149Z.json"


def main():
    w = json.load(open(SEL))["w"]; e = 1e-6
    P = np.load("artifacts/preds_gbdt148.npz"); pi = {n: i for i, n in enumerate(P["eval_names"])}
    S = w[0]*np.log(P["eval_col"]+e) + w[1]*np.log(P["eval_da"]+e) + np.log(.3*P["eval_prog"]+.2*P["eval_comf"]+.5*P["eval_hum"]+e)
    ps = [np.load(f) for f in sorted(glob.glob("artifacts/labels148/validation_[0-9]*.npz"))]
    names = np.concatenate([p["names"] for p in ps]); lab = np.concatenate([p["labels"] for p in ps])
    tr = np.concatenate([p["traj"] for p in ps]).astype(np.float32)
    li = {n: i for i, n in enumerate(names)}
    nps = lambda L: L[..., 0]*L[..., 1]*(.3*L[..., 2]+.2*L[..., 3]+.5*L[..., 4])
    todo = sorted(os.path.basename(f)[:-4] for f in glob.glob("a15_out/nonav/validation/*.npz"))
    todo = [n for n in todo if n in li and n in pi]
    arcs = lambda i: np.sum(np.hypot(*np.diff(tr[i][:, :, :2], axis=1).transpose(2, 0, 1)), axis=1)  # (148,)
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        D = list(ex.map(prep, todo))
        sel = [int(S[pi[d["name"]]].argmax()) for d in D]
        Lsel = [float(arcs(li[d["name"]])[k]) for d, k in zip(D, sel)]
        hyb = np.array(list(ex.map(score, [(d, L) for d, L in zip(D, Lsel)])))
        gb = np.array([nps(lab[li[d["name"]]])[k] for d, k in zip(D, sel)])
        grid = [0, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 77]
        Og = np.array(list(ex.map(score, [(d, L) for d in D for L in grid]))).reshape(len(D), len(grid))
    o148 = np.array([nps(lab[li[d["name"]]]).max() for d in D])
    print(f"{len(D)} clips:  GBDT pick {gb.mean():.4f}   a15 path + GBDT-picked arc {np.nanmean(hyb):.4f}   "
          f"per-clip better {np.mean(hyb > gb + 1e-9):.2f} worse {np.mean(hyb < gb - 1e-9):.2f}")
    print(f"  per-clip max(GBDT, hybrid) {np.maximum(gb, np.nan_to_num(hyb)).mean():.4f}")
    print(f"  oracle 148 {o148.mean():.4f}   oracle 148 + a15 path x 12 arcs {np.maximum(o148, np.nanmax(Og, 1)).mean():.4f}"
          f"   oracle a15 path x 12 arcs alone {np.nanmax(Og, 1).mean():.4f}")


if __name__ == "__main__":
    main()
