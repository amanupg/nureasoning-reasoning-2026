"""Export, per clip, the 63 vocabulary candidates in the KEY-FRAME EGO FRAME plus
ego/route features and (for train/val) the per-candidate sub-metric labels.

Candidates are rebuilt with the SAME function and arguments used to generate the
labels (vocab_labels.build_candidate), and the export asserts the vocabulary
matches, so a label can never be paired with a different trajectory.
Trajectories are subsampled to 11 points (t = 0, 0.5, ..., 5 s).
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "probe")); import _stub_torch  # noqa
sys.path.insert(0, os.path.join(ROOT, "recon", "devkit"))
from concurrent.futures import ProcessPoolExecutor
from nureasoning.planning.benchmark import BenchmarkConfig, select_key_frame_idx, _load_metadata, build_route_line
from nureasoning.common.pickle_io import load_pickle
from nureasoning.submission.format import global_to_ego
import planners
from vocab_labels import VOCAB, build_candidate
from arc_data import features_from
from arc_model import FEATS

CFG = BenchmarkConfig(); SUB = np.arange(0, 51, 5)


def one(cp):
    try:
        meta = _load_metadata(cp); fr = meta.get("frames", [])
        ki = select_key_frame_idx(fr, CFG.key_frame_index)
        ego = load_pickle(os.path.join(cp, fr[ki]["ego_state"]))
        route = build_route_line(fr[ki].get("mission_goal"))
        if route is None or route.is_empty or route.length < 1.0:
            return None
        L0 = planners.arc_length_kinematic(planners._speed(ego), planners._lon_accel(ego))
        T = np.stack([global_to_ego(build_candidate(meta, fr, ki, ego, route, s, l, L0), ego.pose)[SUB]
                      for (s, l) in VOCAB]).astype(np.float32)
        f = features_from(meta, fr, ki, ego)
        return os.path.basename(cp), T, np.array([f.get(k, np.nan) for k in FEATS], np.float32)
    except Exception:
        return None


def main():
    out = {}
    for split in ("train", "validation", "test"):
        root = f"cache/{split}"
        clips = sorted(os.path.join(root, d) for d in os.listdir(root)
                       if os.path.isfile(os.path.join(root, d, "metadata.json")))
        t0 = time.time()
        with ProcessPoolExecutor() as ex:
            res = [r for r in ex.map(one, clips, chunksize=8) if r]
        names = [r[0] for r in res]
        traj = np.stack([r[1] for r in res]).astype(np.float16)
        feat = np.stack([r[2] for r in res])
        labels = None
        lf = {"train": "artifacts/vocab_labels_train.json", "validation": "artifacts/vocab_labels_val.json"}.get(split)
        if lf:
            L = json.load(open(lf))
            assert [tuple(v) for v in L["vocab"]] == [tuple(v) for v in VOCAB], "vocabulary mismatch"
            ymap = {r["clip"]: np.array(r["y"], np.float32) for r in L["rows"]}
            keep = [i for i, n in enumerate(names) if n in ymap]
            names = [names[i] for i in keep]; traj = traj[keep]; feat = feat[keep]
            labels = np.stack([ymap[n] for n in names])
        np.savez_compressed(f"artifacts/cands_{split}.npz", names=np.array(names), traj=traj, feat=feat,
                            vocab=np.array(VOCAB, np.float32),
                            **({"labels": labels} if labels is not None else {}))
        print(f"{split:10} {len(names):5} clips  traj {traj.shape}  feat {feat.shape}  "
              f"labels {None if labels is None else labels.shape}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
