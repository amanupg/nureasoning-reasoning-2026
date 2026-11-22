"""Compare Alpamayo 1.5 against our planners on the pilot clips (validation).

Uses: a15_out/scores_<variant>_validation.jsonl (devkit-scored), labels148 (per-
candidate devkit labels), preds_gbdt148.npz + the selection weights of the v4.1
selections file (to reproduce the GBDT pick on validation), and the cached
ground-truth future for the "how far" comparison.
"""
import glob, json, math, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "probe"), os.path.join(ROOT, "recon", "devkit")]
import _stub_torch  # noqa
from harness import _ctx
from planners import arc_length_kinematic, _speed, _lon_accel

SEL = "artifacts/versioned/selections_test_novision148_20260926T050149Z.json"


def nps(L):
    return L[..., 0] * L[..., 1] * (.3 * L[..., 2] + .2 * L[..., 3] + .5 * L[..., 4])


def main(variant):
    rows = [json.loads(l) for l in open(f"a15_out/scores_{variant}_validation.jsonl")]
    ps = [np.load(f) for f in sorted(glob.glob("artifacts/labels148/validation_*.npz"))]
    names = np.concatenate([p["names"] for p in ps]); lab = np.concatenate([p["labels"] for p in ps])
    li = {n: i for i, n in enumerate(names)}
    P = np.load("artifacts/preds_gbdt148.npz"); pi = {n: i for i, n in enumerate(P["eval_names"])}
    w = json.load(open(SEL))["w"]; cur = json.load(open(SEL))["default_index"]
    e = 1e-6
    S = w[0] * np.log(P["eval_col"] + e) + w[1] * np.log(P["eval_da"] + e) + \
        np.log(.3 * P["eval_prog"] + .2 * P["eval_comf"] + .5 * P["eval_hum"] + e)
    gb = S.argmax(1)

    R = []
    for r in rows:
        c = r["clip"]; m = r["mean"]
        if c not in li or "nps" not in m: continue
        N = nps(lab[li[c]])
        s0 = r["samples"][0]
        meta, fr, ki, ego, route = _ctx(os.path.join("cache/validation", c))
        fut = np.array(ego.trajectory_future, float)
        gt_arc = float(np.sum(np.hypot(*np.diff(np.vstack([[ego.pose["x"], ego.pose["y"]], fut[:50, :2]]), axis=0).T)))
        a15_end = s0["ego11"][-1]; a15_arc = float(np.sum(np.hypot(*np.diff(np.array(s0["ego11"])[:, :2], axis=0).T)))
        R.append(dict(default=N[cur], gbdt=N[gb[pi[c]]] if c in pi else np.nan, a15=s0["nps"], a15mean=m["nps"],
                      a15best=max(s["nps"] for s in r["samples"]), o148=N.max(),
                      o149=max(N.max(), s0["nps"]), gt_arc=gt_arc, a15_arc=a15_arc,
                      kin_arc=arc_length_kinematic(_speed(ego), _lon_accel(ego)),
                      a15_fde=s0["fde"], a15_col=s0["col"], a15_hum=s0["hum"], a15_prog=s0["prog"],
                      stopped=_speed(ego) < 0.5))
    A = {k: np.array([x[k] for x in R], float) for k in R[0]}
    ok = ~np.isnan(A["gbdt"])
    print(f"=== Alpamayo 1.5 [{variant}] vs ours: {len(R)} validation route clips ({ok.sum()} with GBDT pick)")
    for k in ("default", "gbdt", "a15", "a15mean", "a15best", "o148", "o149"):
        print(f"  {k:9} NPS {np.nanmean(A[k][ok]):.4f}")
    print(f"  Alpamayo sample0: col {A['a15_col'].mean():.3f}  prog {A['a15_prog'].mean():.3f}  hum {A['a15_hum'].mean():.3f}  "
          f"FDE p50 {np.median(A['a15_fde']):.2f}")
    for k in ("a15_arc", "kin_arc"):
        err = A[k] - A["gt_arc"]
        print(f"  how-far {k:8}: MAE {np.abs(err).mean():.2f} m  median {np.median(np.abs(err)):.2f} m  bias {err.mean():+.2f} m")
    st = A["stopped"].astype(bool)
    print(f"  stopped clips ({st.sum()}): gbdt {np.nanmean(A['gbdt'][st & ok]):.4f}  a15 {A['a15'][st].mean():.4f}")
    better = (A["a15"] > A["gbdt"] + 1e-9)[ok]; worse = (A["a15"] < A["gbdt"] - 1e-9)[ok]
    print(f"  per clip vs GBDT pick: a15 better {better.mean():.2f}  worse {worse.mean():.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nonav")
