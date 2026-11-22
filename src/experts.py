"""Expert switch on validation: per clip, choose among
  gbdt  : our v4.1 GBDT pick from the 148 candidates
  nuvla : nuVLA epoch-4 trajectory
  a15h  : Alpamayo mean path, re-timed to the GBDT-picked arc length
  a15k  : Alpamayo mean path, re-timed to our kinematic arc length
All scored with the unmodified devkit metric. The switch (one GBDT regressor per
expert predicting NPS - NPS_gbdt) is trained and scored on disjoint DRIVE LOGS
(2 folds x 10 repeats). Validation labels are public; nothing from test.
Also writes artifacts/experts_validation.npz for reuse.
"""
import glob, json, os, random, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from a15_calib import prep, score, CFG
from nureasoning.planning.benchmark import evaluate_clip
from nureasoning.submission.format import global_to_ego
from harness import _ctx
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

SEL = "artifacts/versioned/selections_test_novision148_20260926T050149Z.json"
SUB = np.arange(0, 51, 5)


def score_global(args):
    name, g = args
    r = evaluate_clip(os.path.join("cache/validation", name), CFG, g)
    return np.nan if r.error else r.planning_score


def ego11(args):
    name, g = args
    meta, fr, ki, ego, route = _ctx(os.path.join("cache/validation", name))
    return global_to_ego(g, ego.pose)[SUB]


def main():
    w = json.load(open(SEL))["w"]; e = 1e-6
    P = np.load("artifacts/preds_gbdt148.npz"); pi = {n: i for i, n in enumerate(P["eval_names"])}
    S = w[0]*np.log(P["eval_col"]+e) + w[1]*np.log(P["eval_da"]+e) + np.log(.3*P["eval_prog"]+.2*P["eval_comf"]+.5*P["eval_hum"]+e)
    ps = [np.load(f) for f in sorted(glob.glob("artifacts/labels148/validation_[0-9]*.npz"))]
    names = np.concatenate([p["names"] for p in ps]); lab = np.concatenate([p["labels"] for p in ps])
    tr = np.concatenate([p["traj"] for p in ps]).astype(np.float32); feat = np.concatenate([p["feat"] for p in ps])
    li = {n: i for i, n in enumerate(names)}
    V = np.load("runs_vla/nuvla_e4_validation.npz"); vi = {n: i for i, n in enumerate(V["names"])}
    nps = lambda L: L[..., 0]*L[..., 1]*(.3*L[..., 2]+.2*L[..., 3]+.5*L[..., 4])
    todo = [n for n in P["eval_names"] if n in li and n in vi and os.path.isfile(f"a15_out/nonav/validation/{n}.npz")]
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        D = list(ex.map(prep, todo))
        sel = np.array([int(S[pi[n]].argmax()) for n in todo])
        arc = lambda i, k: float(np.sum(np.hypot(*np.diff(tr[i][k][:, :2], axis=0).T)))
        Lsel = [arc(li[n], k) for n, k in zip(todo, sel)]
        a15h = np.array(list(ex.map(score, [(d, L) for d, L in zip(D, Lsel)])))
        a15k = np.array(list(ex.map(score, [(d, d["kin"]) for d in D])))
        nuv = np.array(list(ex.map(score_global, [(n, V["traj"][vi[n]]) for n in todo])))
        nuv11 = np.array(list(ex.map(ego11, [(n, V["traj"][vi[n]]) for n in todo])))
        a1511 = np.array(list(ex.map(ego11, [(d["name"], d["g"]) for d in D])))
    gb = np.array([nps(lab[li[n]])[k] for n, k in zip(todo, sel)])
    gb11 = np.array([tr[li[n]][k] for n, k in zip(todo, sel)])
    heads = np.stack([P[f"eval_{h}"][[pi[n] for n in todo], sel] for h in ("col", "da", "prog", "comf", "hum")], 1)
    spread = []
    for n in todo:
        z = np.load(f"a15_out/nonav/validation/{n}.npz"); spread.append(np.std(z["xyz"][:, 49, :2], 0).sum())
    X = np.concatenate([feat[[li[n] for n in todo]], heads, gb11[:, -1, :2], nuv11[:, -1, :2], a1511[:, -1, :2],
                        np.hypot(*(nuv11[:, -1, :2] - gb11[:, -1, :2]).T)[:, None],
                        np.hypot(*(a1511[:, -1, :2] - gb11[:, -1, :2]).T)[:, None],
                        np.array(spread)[:, None], np.array([[d["arc"], d["kin"]] for d in D])], 1)
    Y = {"gbdt": gb, "nuvla": np.nan_to_num(nuv), "a15h": np.nan_to_num(a15h), "a15k": np.nan_to_num(a15k)}
    np.savez_compressed("artifacts/experts_validation.npz", names=np.array(todo), X=X, **Y)
    print(f"{len(todo)} validation clips: " + "  ".join(f"{k} {v.mean():.4f}" for k, v in Y.items()))
    print(f"  oracle over experts {np.max(np.stack(list(Y.values())), 0).mean():.4f}")
    log_of = lambda c: "_".join(c.split("_")[:2]); logs = sorted({log_of(n) for n in todo})
    G = np.array([log_of(n) for n in todo]); res = []
    for seed in range(10):
        random.Random(seed).shuffle(logs); h = set(logs[:len(logs) // 2])
        for fit in (np.isin(G, list(h)), ~np.isin(G, list(h))):
            ev = ~fit; gain = {}
            for k in ("nuvla", "a15h", "a15k"):
                m = HGB(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, random_state=0)
                m.fit(X[fit], (Y[k] - Y["gbdt"])[fit]); gain[k] = m.predict(X[ev])
            Gm = np.stack([np.zeros(ev.sum())] + [gain[k] for k in ("nuvla", "a15h", "a15k")], 1)
            ch = Gm.argmax(1); M = np.stack([Y[k][ev] for k in ("gbdt", "nuvla", "a15h", "a15k")], 1)
            res.append((M[np.arange(len(ch)), ch].mean(), Y["gbdt"][ev].mean(), np.bincount(ch, minlength=4) / len(ch)))
    r = np.array([x[:2] for x in res])
    print(f"  switch held-out (20 folds): {r[:, 0].mean():.4f} vs gbdt {r[:, 1].mean():.4f}  gain {np.mean(r[:, 0] - r[:, 1]):+.4f} "
          f"(sd {np.std(r[:, 0] - r[:, 1]):.4f}); choice shares gbdt/nuvla/a15h/a15k {np.mean([x[2] for x in res], 0).round(2)}")


if __name__ == "__main__":
    main()
