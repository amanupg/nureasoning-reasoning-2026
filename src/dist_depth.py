"""Metric depth (Depth Anything 3 Metric) inside the stated box of distance_from_2d_bbox questions.

Per (clip, camera) at frame 100: DA3METRIC-LARGE on the cached image (img/<split>/...), metric depth =
f * net / 300 with f = the clip's TRUE focal length rescaled to the processed resolution. Per question:
depth percentiles inside the box (inner 60% x 60% and full box) -> ego-frame distance along the ray
through the box centre; plus ground-plane scale probes (model depth / exact ground-plane depth of the
same pixels, in bottom strips outside the box) for an optional per-image scale correction.
-> <out>_s{i}of{n}.json {question_id: {...}}
usage: python src/dist_depth.py <split> <out_prefix> [--res 1008] [--shard i --nshards n]
"""
import argparse, json, math, os, sys
import numpy as np
import torch
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "ext", "da3", "src")]
import camgeo
from dist_mono import questions


def ego_dist(c, u, v, z):
    pc = np.linalg.inv(c["K"]) @ np.array([u / 1000 * c["W"], v / 1000 * c["H"], 1.0])
    P = c["R"] @ (pc * z / pc[2]) + c["t"]
    return math.hypot(P[0], P[1])


def ground_depth(c, u, v, gz=0.0):
    """Camera z of the ground-plane (ego z = gz) point seen at (u, v) in 0..1000, or nan."""
    pc = np.linalg.inv(c["K"]) @ np.array([u / 1000 * c["W"], v / 1000 * c["H"], 1.0])
    d = c["R"] @ pc
    if d[2] >= -1e-4: return float("nan")
    s = (gz - c["t"][2]) / d[2]
    return s * pc[2]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("split"); ap.add_argument("out")
    ap.add_argument("--res", type=int, default=1008); ap.add_argument("--img", default=os.environ.get("IMG_ROOT", "img"))
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    from unittest.mock import MagicMock                   # export/app-only deps of the DA3 package, unused here
    for m in ("pycolmap", "open3d", "evo", "evo.core", "evo.core.trajectory", "pillow_heif", "gradio", "fastapi",
              "fastapi.responses", "uvicorn", "typer", "sklearn"):
        try: __import__(m)
        except ImportError: sys.modules[m] = MagicMock()
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to("cuda").eval()
    from transformers import Sam2Model, Sam2Processor            # object-only pixels: SAM 2.1 mask from the box
    sam = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to("cuda").eval()
    sproc = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
    Q = questions(a.split); groups = {}
    for x in Q: groups.setdefault((x["clip"], x["cam"]), []).append(x)
    keys = sorted(groups)[a.shard::a.nshards]
    out_path = f"{a.out}_s{a.shard}of{a.nshards}.json"
    res = json.load(open(out_path)) if os.path.isfile(out_path) else {}
    print(f"{len(Q)} questions, {len(keys)} images in shard, {len(res)} done", flush=True)
    for n, (clip, cam) in enumerate(keys):
        G = [x for x in groups[(clip, cam)] if x["q"]["question_id"] not in res]
        if not G: continue
        meta = json.load(open(f"cache/{a.split}/{clip}/metadata.json")); rig = camgeo.Rig(meta)
        rel = meta["frames"][100]["sensors"]["cameras"].get(cam); p = f"{a.img}/{a.split}/{clip}/{rel}"
        if cam not in rig.cams or not rel or not os.path.isfile(p):
            for x in G: res[x["q"]["question_id"]] = {"missing": True}
            continue
        c = rig.cams[cam]
        im = Image.open(p).convert("RGB"); IW, IH = im.size
        with torch.inference_mode():
            pred = model.inference([im], process_res=a.res)
            boxes = [[x["box"][0] * IW / 1000, x["box"][1] * IH / 1000, x["box"][2] * IW / 1000, x["box"][3] * IH / 1000] for x in G]
            inp = sproc(images=im, input_boxes=[boxes], return_tensors="pt").to("cuda")
            so = sam(**inp, multimask_output=False)
            masks = sproc.post_process_masks(so.pred_masks.cpu(), inp["original_sizes"].cpu())[0][:, 0].numpy()
        net = np.asarray(pred.depth[0], np.float32); h, w = net.shape
        f = 0.5 * (c["K"][0, 0] * w / c["W"] + c["K"][1, 1] * h / c["H"])
        D = f * net / 300.0
        # ground-plane scale probes (outside every box of this image)
        mask = np.ones((h, w), bool)
        for x in G:
            b = x["box"]; mask[int(b[1] / 1000 * h):int(math.ceil(b[3] / 1000 * h)), int(b[0] / 1000 * w):int(math.ceil(b[2] / 1000 * w))] = False
        probes = {}
        for gz in (0.0, -0.35):
          for lo, hi in ((0.70, 0.80), (0.80, 0.90), (0.90, 0.98)):
            r = []
            for vv in np.linspace(lo * 1000, hi * 1000, 6):
                for uu in np.linspace(300, 700, 9):
                    i, j = min(h - 1, int(vv / 1000 * h)), min(w - 1, int(uu / 1000 * w))
                    gd = ground_depth(c, uu, vv, gz)
                    if mask[i, j] and np.isfinite(gd) and 1.0 < gd < 60: r.append(float(D[i, j] / gd))
            probes[f"z{gz}_{lo:.2f}"] = [float(np.median(r)), len(r)] if r else [float("nan"), 0]
        for gi, x in enumerate(G):
            b = x["box"]; uc, vc = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            def crop(fr):
                du, dv = (b[2] - b[0]) * (1 - fr) / 2, (b[3] - b[1]) * (1 - fr) / 2
                i0, i1 = int((b[1] + dv) / 1000 * h), int(math.ceil((b[3] - dv) / 1000 * h))
                j0, j1 = int((b[0] + du) / 1000 * w), int(math.ceil((b[2] - du) / 1000 * w))
                return D[max(0, i0):max(i0 + 1, min(h, i1)), max(0, j0):max(j0 + 1, min(w, j1))].ravel()
            st = {}
            for fr in (0.6, 1.0):
                v = crop(fr)
                for pc in (10, 25, 50):
                    z = float(np.percentile(v, pc)); st[f"f{fr}_p{pc}"] = [z, ego_dist(c, uc, vc, z)]
            mk = np.asarray(Image.fromarray(masks[gi].astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 0
            v = D[mk]
            if v.size >= 3:
                ys, xs = np.nonzero(mk); mu, mv = 1000 * xs.mean() / w, 1000 * ys.mean() / h
                for pc in (10, 25, 50):
                    z = float(np.percentile(v, pc)); st[f"mask_p{pc}"] = [z, ego_dist(c, mu, mv, z)]
            st["mask_frac"] = [float(mk.mean()), float(v.size)]
            res[x["q"]["question_id"]] = dict(clip=clip, cam=cam, cat=x["cat"], box=b, depth=st, probes=probes,
                                              choices=x["q"]["choices"], answer=x["q"].get("answer"))
        if n % 50 == 0:
            json.dump(res, open(out_path, "w")); print(f"  {n}/{len(keys)}", flush=True)
    json.dump(res, open(out_path, "w")); print("done", len(res), flush=True)


if __name__ == "__main__":
    main()
