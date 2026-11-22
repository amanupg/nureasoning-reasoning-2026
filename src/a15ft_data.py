"""nuReasoning -> Alpamayo 1.5 fine-tuning samples (drop-in for alpamayo-recipes'
PAIDataset: same keys, same shapes, same preprocessing hook).

One sample per reasoning-annotated frame r of a train clip, if
  * r >= 15 (16-step ego history) and 64 future ego states exist (6.4 s),
  * all 12 images exist: cameras front_left / front / front_right (Alpamayo slots
    0 / 1 / 2, as in our zero-shot inference) at frames r-3 .. r.
Inputs are built with EXACTLY the math of src/a15_infer.py (history = 15 points of
ego.trajectory_history + current pose, t0 ego frame, z = 0, yaw-only rotations), so
training and inference see identical inputs. Images are resized to a fixed size
(default 896x592) for both training and inference.
Future: per-frame ego poses r+1..r+64 (10 Hz, 6.4 s), same t0 frame.
"""
import json, math, os
from typing import Any
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CAMS = [("front_left", 0), ("front", 1), ("front_right", 2)]
ROOT = os.environ.get("NUREASONING_ROOT", "/path/to/cluster/scratch/nureasoning")


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _load_pickle(p):
    import sys
    sys.path.insert(0, os.path.join(ROOT, "recon", "devkit"))
    from nureasoning.common.pickle_io import load_pickle
    return load_pickle(p)


def build_sample(cache_clip, img_roots, r, size=(896, 592), with_future=True):
    """Alpamayo input dict for frame r of one clip (None if anything is missing)."""
    fr = json.load(open(os.path.join(cache_clip, "metadata.json")))["frames"]
    if r < 15 or r >= len(fr):
        return None
    ego = _load_pickle(os.path.join(cache_clip, fr[r]["ego_state"]))
    hist = np.array(getattr(ego, "trajectory_history", None) or [], float)
    if len(hist) < 15:
        return None
    fut = None
    if with_future:
        # 6.4 s future = per-frame ego poses r+1..r+64 (the stored trajectory_future holds
        # only 50 points; per-frame poses reproduce it exactly, max diff 0.0000 m)
        if r + 64 >= len(fr):
            return None
        fut = []
        for k in range(r + 1, r + 65):
            pk = _load_pickle(os.path.join(cache_clip, fr[k]["ego_state"])).pose
            fut.append([pk["x"], pk["y"], pk["yaw"]])
        fut = np.array(fut, float)
    imgs, ts = [], []
    for cam, _ in CAMS:
        cts = []
        for k in (r - 3, r - 2, r - 1, r):
            rel = fr[k]["sensors"]["cameras"][cam]
            p = next((os.path.join(d, rel) for d in img_roots if os.path.isfile(os.path.join(d, rel))), None)
            if p is None:
                return None
            im = Image.open(p).convert("RGB")
            if im.size != size:
                im = im.resize(size, Image.BILINEAR)
            imgs.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1))
            cts.append(int(fr[k]["timestamp_us"]))
        ts.append(cts)
    x0, y0, yaw0 = ego.pose["x"], ego.pose["y"], ego.pose["yaw"]
    R0 = rotz(yaw0)[:2, :2]

    def local(P):
        xy = (P[:, :2] - [x0, y0]) @ R0
        return np.column_stack([xy, np.zeros(len(P))]), np.stack([rotz(a - yaw0) for a in P[:, 2]])

    hx, hr = local(np.vstack([hist[-15:], [x0, y0, yaw0]]))
    at = torch.tensor(ts, dtype=torch.int64)
    out = {
        "image_frames": torch.stack(imgs).view(len(CAMS), 4, 3, size[1], size[0]),
        "camera_indices": torch.tensor([c for _, c in CAMS], dtype=torch.int64),
        # (1, 1, T, 3) exactly like load_physical_aiavdataset; datasets squeeze(0) -> (1, T, 3)
        "ego_history_xyz": torch.from_numpy(hx).float()[None, None],
        "ego_history_rot": torch.from_numpy(hr).float()[None, None],
        "absolute_timestamps": at,
        "relative_timestamps": (at - at.min()).float() * 1e-6,
        "t0_us": int(fr[r]["timestamp_us"]),
        "clip_id": os.path.basename(cache_clip),
    }
    if with_future:
        fx, frt = local(fut[:64])
        out["ego_future_xyz"] = torch.from_numpy(fx).float()[None, None]
        out["ego_future_rot"] = torch.from_numpy(frt).float()[None, None]
    return out


class NuReasoningA15Dataset(Dataset):
    """Samples = (clip, reasoning frame) for the clips of one half (or all)."""

    def __init__(self, split: str = "train", half: str | None = None, img_roots: list[str] | None = None,
                 index_path: str | None = None, model_config: Any | None = None,
                 vla_preprocess_args: dict | None = None, max_samples: int = 0, size=(896, 592),
                 **unused: Any):   # PAIDataset fields from sft_base (local_dir, chunk_ids, ...) are ignored
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        self.cache = os.path.join(ROOT, "cache", split)
        self.img_roots = [os.path.join(ROOT, d, split) for d in (img_roots or ["img_full", "img_a15ft", "img_a15"])]
        self.size = tuple(size)
        idx = index_path or os.path.join(ROOT, f"artifacts/a15ft_index_{split}.json")
        rows = json.load(open(idx))                  # [[clip, r], ...] precomputed (all files present)
        if half is not None:
            keep = set(json.load(open(os.path.join(ROOT, "artifacts/vla_halves.json")))[half])
            rows = [x for x in rows if x[0] in keep]
        self.rows = rows[:max_samples] if max_samples else rows
        if isinstance(model_config, dict):
            model_config = OmegaConf.create(model_config)
        self.pre = instantiate(vla_preprocess_args, model_config=model_config) if vla_preprocess_args else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        clip, r = self.rows[i][:2]
        s = build_sample(os.path.join(self.cache, clip), [os.path.join(d, clip) for d in self.img_roots], r, self.size)
        if s is None:
            return None
        if len(self.rows[i]) > 2:                    # stage-1 index (a15s1_index.py): reasoning trace as CoT target
            s["cot"] = self.rows[i][2]
        for k in list(s):
            if k.startswith("ego_"):
                s[k] = s[k].squeeze(0)
        if self.pre is not None:
            s["tokenized_data"] = self.pre(data=s)
        return s
