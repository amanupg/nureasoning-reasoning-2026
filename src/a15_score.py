"""Score Alpamayo 1.5 trajectories with the UNMODIFIED devkit metric (evaluate_clip).

For each clip: every sample, plus the sample mean. Local t0-frame (64 steps at
t0+0.1..6.4 s) -> first 50 steps, prepend t0 -> global (51, 3) [x, y, yaw].
Also exports each trajectory in the same 11-point ego-frame format as labels148,
so it can be appended to the candidate set.
"""
import argparse, glob, json, math, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "probe"), os.path.join(ROOT, "recon", "devkit")]
import _stub_torch  # noqa: F401,E402
from nureasoning.planning.benchmark import BenchmarkConfig, evaluate_clip          # noqa: E402
from nureasoning.submission.format import global_to_ego                             # noqa: E402
from harness import _ctx                                                            # noqa: E402

CFG = BenchmarkConfig(); SUB = np.arange(0, 51, 5)


def to_global(xyz, rot, pose):
    x0, y0, yaw0 = pose
    loc = np.vstack([[0.0, 0.0], xyz[:50, :2]])
    yl = np.concatenate([[0.0], np.arctan2(rot[:50, 1, 0], rot[:50, 0, 0])])
    c, s = math.cos(yaw0), math.sin(yaw0)
    gx = x0 + c * loc[:, 0] - s * loc[:, 1]
    gy = y0 + s * loc[:, 0] + c * loc[:, 1]
    return np.column_stack([gx, gy, yaw0 + yl])


def one(args):
    f, split = args
    clip = os.path.basename(f)[:-4]; cp = os.path.join("cache", split, clip)
    z = np.load(f)
    xyz, rot, pose = z["xyz"], z["rot"], tuple(z["pose"])
    meta, fr, ki, ego, route = _ctx(cp)
    assert abs(ego.pose["x"] - pose[0]) < 1e-6 and abs(ego.pose["y"] - pose[1]) < 1e-6
    trajs = [to_global(xyz[k], rot[k], pose) for k in range(len(xyz))]
    m = xyz.mean(0); ang = np.arctan2(rot[:, :, 1, 0], rot[:, :, 0, 0])
    mr = np.zeros_like(rot[0]); a = np.arctan2(np.sin(ang).mean(0), np.cos(ang).mean(0))
    mr[:, 0, 0] = mr[:, 1, 1] = np.cos(a); mr[:, 1, 0] = np.sin(a); mr[:, 0, 1] = -np.sin(a); mr[:, 2, 2] = 1
    trajs.append(to_global(m, mr, pose))
    out = []
    for g in trajs:
        r = evaluate_clip(cp, CFG, g)
        if r.error:
            out.append({"error": r.error}); continue
        out.append({"nps": r.planning_score, "col": r.s_collision, "da": r.s_driveable,
                    "prog": r.s_progress, "comf": r.s_comfort, "hum": r.s_human,
                    "fde": r.details_human.get("fde"), "ego11": global_to_ego(g, ego.pose)[SUB].tolist()})
    return {"clip": clip, "nav": str(z["nav"]), "samples": out[:-1], "mean": out[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--variant", default="nonav")
    ap.add_argument("--root", default="a15_out")
    a = ap.parse_args()
    fs = sorted(glob.glob(f"{a.root}/{a.variant}/{a.split}/*.npz"))
    out = f"{a.root}/scores_{a.variant}_{a.split}.jsonl"
    with open(out, "w") as fh, ProcessPoolExecutor(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as ex:
        R = list(ex.map(one, [(f, a.split) for f in fs], chunksize=2))
        for r in R: fh.write(json.dumps(r) + "\n")
    ok = [r for r in R if "nps" in r["mean"]]
    A = lambda k, key="mean": np.mean([r[key][k] for r in ok]) if key == "mean" else \
        np.mean([r["samples"][0][k] for r in ok])
    print(f"{a.variant} {a.split}: n={len(ok)}")
    for key in ("sample0", "mean"):
        print(f"  {key:8} NPS {A('nps', key):.4f}  col {A('col', key):.3f}  da {A('da', key):.3f}  "
              f"prog {A('prog', key):.3f}  comf {A('comf', key):.3f}  hum {A('hum', key):.3f}  "
              f"FDE p50 {np.median([(r['mean'] if key == 'mean' else r['samples'][0])['fde'] for r in ok]):.2f}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
