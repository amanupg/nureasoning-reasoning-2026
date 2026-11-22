"""Frozen DINOv3 features for the key-frame camera images.

Per clip: 2 frames (key frame idx 100, and idx 90 = 1 s earlier) x 8 cameras.
Each image -> DINOv3 at 512x336 (keeps the 2816x1856 aspect, multiples of 16)
-> CLS token + patch grid average-pooled to 6x8. Stored fp16 as
feat/<model>/<split>/<clip>.npy with shape [2, 8, 1+48, D].

Why this shape: the literature's largest clean vision effect is PRETRAINING
(random 70.1 -> DINOv2 90.0 PDMS), not backbone size; DrivoR reaches 94.6
PDMS camera-only on a ViT-S. A coarse spatial grid keeps "is something in my
lane ahead" answerable while staying small enough to train a scorer on 10k clips.
Never overwrites: existing .npy files are skipped.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

CAMS = ["back", "back_left", "back_right", "front", "front_left", "front_right", "left", "right"]
W, H = 512, 336
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class Clips(Dataset):
    def __init__(self, rows, img_root, out_dir):
        self.rows = [r for r in rows
                     if os.path.isfile(os.path.join(img_root, r["clip"], "_done.json"))
                     and not os.path.isfile(os.path.join(out_dir, r["clip"] + ".npy"))]
        self.img_root = img_root

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ims = []
        for m in r["members"]:                     # 16 = 2 frames x 8 cams, fixed order
            p = os.path.join(self.img_root, r["clip"], m.split("/", 1)[1])
            im = Image.open(p).convert("RGB").resize((W, H), Image.BILINEAR)
            t = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
            ims.append((t - MEAN) / STD)
        return r["clip"], torch.stack(ims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--members", default="src/image_members.json")
    ap.add_argument("--img", default="img")
    ap.add_argument("--out", default="feat")
    ap.add_argument("--bs", type=int, default=4)          # clips per batch (x16 images)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    from transformers import AutoModel
    dev = "cuda"
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.float16).to(dev).eval()
    nreg = int(getattr(model.config, "num_register_tokens", 0))
    tag = a.model.split("/")[-1]
    out_dir = os.path.join(a.out, tag, a.split); os.makedirs(out_dir, exist_ok=True)

    rows = json.load(open(a.members))[a.split]
    for r in rows:                                   # canonical order: frame100 cams, frame90 cams
        assert len(r["members"]) == 16, (r["clip"], len(r["members"]))
    ds = Clips(rows, os.path.join(a.img, a.split), out_dir)
    dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, pin_memory=True)
    print(f"[feat] {tag} split={a.split} todo={len(ds)} clips  registers={nreg}", flush=True)

    json.dump({"model": a.model, "size": [W, H], "grid": [6, 8], "cams": CAMS,
               "frames": [100, 90], "layout": "[frame, cam, 1+48 tokens, D]",
               "script": os.path.abspath(__file__),
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
              open(os.path.join(out_dir, "_meta.json"), "w"))
    t0 = time.time(); n = 0
    with torch.inference_mode():
        for clips, x in dl:
            B = x.shape[0]
            x = x.view(B * 16, 3, H, W).to(dev, non_blocking=True).half()
            h = model(pixel_values=x).last_hidden_state          # [B*16, 1+reg+N, D]
            cls = h[:, 0:1]
            patch = h[:, 1 + nreg:]
            gh, gw = H // 16, W // 16
            patch = patch.transpose(1, 2).reshape(B * 16, -1, gh, gw)
            grid = F.adaptive_avg_pool2d(patch.float(), (6, 8)).flatten(2).transpose(1, 2)
            f = torch.cat([cls.float(), grid], 1).view(B, 2, 8, 49, -1).half().cpu().numpy()
            for c, arr in zip(clips, f):
                tmp = os.path.join(out_dir, c + ".npy.tmp")
                np.save(tmp, arr); os.replace(tmp + ".npy" if os.path.exists(tmp + ".npy") else tmp,
                                              os.path.join(out_dir, c + ".npy"))
            n += B
            if n % 200 < B:
                print(f"  {n}/{len(ds)}  {n/(time.time()-t0):.1f} clips/s", flush=True)
    print(f"[feat] DONE {n} clips in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
