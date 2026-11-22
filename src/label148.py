"""Label every clip's 148 candidates with the unmodified devkit scorer AND export
their ego-frame trajectories in the same pass (so labels and trajectories can
never be mismatched). Sharded for Slurm arrays; one .npz per shard."""
import argparse, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "probe")); import _stub_torch  # noqa
sys.path.insert(0, os.path.join(ROOT, "recon", "devkit"))
from concurrent.futures import ProcessPoolExecutor
from nureasoning.planning.benchmark import BenchmarkConfig, evaluate_clip, select_key_frame_idx, _load_metadata, build_route_line
from nureasoning.common.pickle_io import load_pickle
from nureasoning.submission.format import global_to_ego
from vocab148 import VOCAB148, build
from arc_data import features_from
from arc_model import FEATS

CFG = BenchmarkConfig(); SUB = np.arange(0, 51, 5)


def one(args):
    cp, with_labels, kf = args
    cfg = CFG if kf == CFG.key_frame_index else BenchmarkConfig(key_frame_index=kf)
    try:
        meta = _load_metadata(cp); fr = meta.get("frames", []); ki = select_key_frame_idx(fr, cfg.key_frame_index)
        if ki != kf: return None
        ego = load_pickle(os.path.join(cp, fr[ki]["ego_state"])); route = build_route_line(fr[ki].get("mission_goal"))
        if route is None or route.is_empty or route.length < 1.0: return None
        T, Y = [], []
        for spec in VOCAB148:
            g = build(meta, fr, ki, ego, route, spec)
            T.append(global_to_ego(g, ego.pose)[SUB])
            if with_labels:
                r = evaluate_clip(cp, cfg, g)
                if r.error: return None
                Y.append([r.s_collision, r.s_driveable, r.s_progress, r.s_comfort, r.s_human])
        f = features_from(meta, fr, ki, ego)
        return (os.path.basename(cp), np.array(T, np.float16), np.array([f.get(k, np.nan) for k in FEATS], np.float32),
                np.array(Y, np.float32) if with_labels else None)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default="artifacts/labels148")
    ap.add_argument("--key-frame", type=int, default=CFG.key_frame_index,
                    help="label a non-key frame as an extra TRAINING sample (train split only)")
    a = ap.parse_args()
    root = f"cache/{a.split}"
    clips = sorted(os.path.join(root, d) for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "metadata.json")))
    clips = clips[a.shard::a.nshards]
    with_labels = a.split != "test"
    assert a.key_frame == CFG.key_frame_index or a.split == "train", "extra frames are for training only"
    t0 = time.time()
    with ProcessPoolExecutor(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))) as ex:
        R = [r for r in ex.map(one, [(c, with_labels, a.key_frame) for c in clips], chunksize=2) if r]
    os.makedirs(a.out, exist_ok=True)
    kf = "" if a.key_frame == CFG.key_frame_index else f"_kf{a.key_frame:03d}"
    out = f"{a.out}/{a.split}{kf}_{a.shard:03d}of{a.nshards:03d}.npz"
    np.savez_compressed(out, names=np.array([r[0] for r in R]), traj=np.stack([r[1] for r in R]),
                        feat=np.stack([r[2] for r in R]),
                        vocab=np.array([[1.0 if k == "abs" else 0.0, v, l] for k, v, l in VOCAB148], np.float32),
                        **({"labels": np.stack([r[3] for r in R])} if with_labels else {}))
    print(f"{a.split} shard {a.shard}/{a.nshards}: {len(R)}/{len(clips)} clips -> {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
