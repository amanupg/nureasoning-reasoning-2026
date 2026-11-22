"""DINOv3 embeddings of box crops for the box-conditioned 3D regressor.
Per sample: key-frame crop (box enlarged 1.3x, square, 224 px) and the same region in the older frame
(enlarged 1.6x, to contain the object's motion) -> [CLS, mean patch] each (2 x 768) -> 3,072 floats.
-> <jsonl stem>.npz {emb (N, 3072) float16, ok (N,) bool}
usage: python src/box3d_embed.py <samples.jsonl> [--model facebook/dinov3-vitb16-pretrain-lvd1689m]
"""
import argparse, glob, json, os
import numpy as np
import torch
from PIL import Image
KEY = {"front": "CAM_M_F", "front_left": "CAM_M_L0", "front_right": "CAM_M_R0", "left": "CAM_M_L1",
       "right": "CAM_M_R1", "back_left": "CAM_M_L2", "back_right": "CAM_M_R2", "back": "CAM_M_B"}
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def frames(split, clip, cam):
    fs = sorted(glob.glob(f"{os.environ.get('IMG_ROOT', 'img')}/{split}/{clip}/cameras/{KEY[cam]}/*.jpg"),
                key=lambda p: int(os.path.basename(p).rsplit("_", 1)[1].split(".")[0]))
    return (fs[-2], fs[-1]) if len(fs) >= 2 else None


def crop(im, box, k):
    W, H = im.size; x1, y1, x2, y2 = [b / 1000 for b in box]
    cx, cy = (x1 + x2) / 2 * W, (y1 + y2) / 2 * H; s = max((x2 - x1) * W, (y2 - y1) * H, 8) * k / 2
    c = im.crop((int(cx - s), int(cy - s), int(cx + s), int(cy + s))).resize((224, 224), Image.BILINEAR)
    return (np.asarray(c, np.float32) / 255 - MEAN) / STD


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("jsonl"); ap.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--bs", type=int, default=128); a = ap.parse_args()
    from transformers import AutoModel
    m = AutoModel.from_pretrained(a.model, dtype=torch.float16).cuda().eval()
    nreg = int(getattr(m.config, "num_register_tokens", 0))
    S = [json.loads(l) for l in open(a.jsonl)]
    emb = np.zeros((len(S), 3072), np.float16); ok = np.zeros(len(S), bool)
    order = sorted(range(len(S)), key=lambda i: (S[i]["split"], S[i]["clip"], S[i]["cam"]))
    cache_key, ims = None, None; buf, idx = [], []

    def flush():
        if not buf: return
        x = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).cuda().half()
        with torch.no_grad():
            h = m(pixel_values=x).last_hidden_state
        f = torch.cat([h[:, 0], h[:, 1 + nreg:].mean(1)], 1).float().cpu().numpy()
        for j, i in enumerate(idx[::2]):
            emb[i] = np.concatenate([f[2 * j], f[2 * j + 1]]); ok[i] = True
        buf.clear(); idx.clear()
    for n, i in enumerate(order):
        s = S[i]; key = (s["split"], s["clip"], s["cam"])
        if key != cache_key:
            fr = frames(*key); cache_key = key
            ims = (Image.open(fr[0]).convert("RGB"), Image.open(fr[1]).convert("RGB")) if fr else None
        if ims is None: continue
        buf += [crop(ims[1], s["box"], 1.3), crop(ims[0], s["box"], 1.6)]; idx += [i, i]
        if len(buf) >= a.bs: flush()
        if n % 10000 == 0: print(f"  {n}/{len(S)}", flush=True)
    flush()
    out = a.jsonl.rsplit(".", 1)[0] + ".npz"
    np.savez_compressed(out, emb=emb, ok=ok); print(f"{ok.sum()}/{len(S)} embedded -> {out}", flush=True)


if __name__ == "__main__":
    main()
