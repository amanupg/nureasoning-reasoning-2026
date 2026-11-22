"""Trajectories from a fine-tuned Alpamayo 1.5 checkpoint (stage-2 SFT), sampled the way
the recipe's ReasoningSampler does, with inputs built by src/a15ft_data.build_sample at
the key frame (100) - identical preprocessing to training (896x592 images).

Output format = src/a15_infer.py (xyz (S,64,3), rot (S,64,3,3), pose), so
src/label_a15.py and the chooser reuse it unchanged:
  <out>/<split>/<clip>.npz
Out-of-fold use: the model trained on half h0 predicts train clips of h1 (and v.v.).
"""
import argparse, json, os, sys, time
from functools import partial
import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a15ft_data import build_sample, ROOT
from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1
from alpamayo.processor.qwen_processor import collate_fn_from_model_config

REC = os.path.join(ROOT, "ext/alpamayo-recipes/recipes/alpamayo1_5_sft/configs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--clips", default="", help="json list of clips (default: all with key-frame images)")
    ap.add_argument("--half", default="", help="restrict to this half from artifacts/vla_halves.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    img_roots = [os.path.join(ROOT, d, a.split) for d in ("img_a15", "img_full", "img_a15ft", "img")]
    if a.clips:
        clips = json.load(open(a.clips))
    else:
        clips = sorted(c for c in os.listdir(os.path.join(ROOT, "img_a15", a.split))
                       if os.path.isfile(os.path.join(ROOT, "img_a15", a.split, c, "_done.json")))
    if a.half:
        keep = set(json.load(open(os.path.join(ROOT, "artifacts/vla_halves.json")))[a.half])
        clips = [c for c in clips if c in keep]
    clips = clips[a.shard::a.nshards]
    if a.limit: clips = clips[:a.limit]
    out = os.path.join(a.out, a.split); os.makedirs(out, exist_ok=True)
    todo = [c for c in clips if not os.path.isfile(os.path.join(out, c + ".npz"))]
    print(f"[a15ft] {a.split}: {len(todo)}/{len(clips)} to do", flush=True)
    if not todo: return

    model = TrainableAlpamayoR1.from_pretrained(a.ckpt, dtype=torch.bfloat16).cuda().eval()
    pre_cfg = OmegaConf.load(f"{REC}/vla_processor/default.yaml"); pre_cfg["generation_mode"] = True
    pre = instantiate(pre_cfg, model_config=model.config)
    collate = partial(collate_fn_from_model_config, model_config=model.config, chat_template_version="r1_5")

    def to(x):
        if torch.is_tensor(x): return x.cuda()
        if isinstance(x, dict): return {k: to(v) for k, v in x.items()}
        return x
    t0 = time.time()
    for n, c in enumerate(todo):
        s = build_sample(os.path.join(ROOT, "cache", a.split, c), [os.path.join(d, c) for d in img_roots], 100,
                         with_future=False)
        if s is None:
            print("  skip", c, flush=True); continue
        pose = None
        for k in list(s):
            if k.startswith("ego_"): s[k] = s[k].squeeze(0)
        s["tokenized_data"] = pre(data=s)
        batch = to(collate([s]))
        torch.cuda.manual_seed_all(42)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            xyz, rot = model.sample_trajectories_from_data(data=batch, num_traj_samples=a.samples, num_traj_sets=1,
                                                            top_p=0.98, temperature=0.6, max_generation_length=256)[:2]
        from a15ft_data import _load_pickle
        fr = json.load(open(os.path.join(ROOT, "cache", a.split, c, "metadata.json")))["frames"]
        ego = _load_pickle(os.path.join(ROOT, "cache", a.split, c, fr[100]["ego_state"]))
        pose = (ego.pose["x"], ego.pose["y"], ego.pose["yaw"])
        np.savez_compressed(os.path.join(out, c + ".npz"), xyz=xyz[0, 0].float().cpu().numpy(),
                            rot=rot[0, 0].float().cpu().numpy(), pose=np.array(pose), nav=np.array(""),
                            cot=np.array([""]))
        if n < 2 or (n + 1) % 50 == 0:
            print(f"  [{n+1}/{len(todo)}] {(time.time()-t0)/(n+1):.1f}s/clip end={xyz[0,0,0,-1,:2].tolist()}", flush=True)
    print(f"[a15ft] done {len(todo)} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
