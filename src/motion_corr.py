"""Object motion from two frames by DINOv3 correspondence + calibration (no training).

For a question that states an object's camera, 2D box (frame 100) and ego-frame
position: take the DINOv3 patch features inside the box at frame 100, slide that
template over the same camera's frame 90 (1 s earlier) and take the best match.
The matched box bottom is lifted to the ground plane with the clip's calibration,
moved into the frame-100 ego frame with the ego poses, and differenced against the
stated position -> a velocity estimate. Pure measurement; a small classifier on
top (trained on TRAIN questions) turns it into the answer.

Outputs artifacts/motion_<split>.json: {question_id: features}.
"""
import argparse, glob, json, math, os, re, sys, time
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "recon", "devkit")]
import camgeo
from nureasoning.common.pickle_io import load_pickle

SUBS = {"future_motion_label"}
RE_BOX = re.compile(r'normalized 2D box \[([-\d\s,\.]+)\]')
RE_CAM = re.compile(r'in the ([a-z ]+?) camera')
RE_POS = re.compile(r'\[x, y\]\s*=\s*\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]')
W, H = 1024, 672                      # 64 x 42 patches of 16 px
GW, GH = W // 16, H // 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
SEARCH = 16                           # patches
SCALES = (0.8, 0.9, 1.0, 1.12, 1.25)  # frame-90 enlargement factors (object k x farther at 90 if best k > 1)


def questions(split):
    if split == "test":
        files = glob.glob("cache/test/*/reasoning_questions.json")
        clip_of = lambda f: f.split("/")[2]
    else:
        files = glob.glob(f"{'vqa_val' if split == 'validation' else 'vqa_train'}/*/*_vqa.json")
        clip_of = lambda f: f.split("/")[1]
    out = []
    for f in files:
        J = json.load(open(f)); qs = J if isinstance(J, list) else J.get("questions", [])
        for q in qs:
            if q.get("subcategory") not in SUBS: continue
            t = q["question"]; mb, mc, mp = RE_BOX.search(t), RE_CAM.search(t), RE_POS.search(t)
            if not (mb and mc and mp): continue
            box = [float(v) for v in mb.group(1).split(",")]
            if len(box) != 4: continue
            out.append(dict(qid=q["question_id"], clip=clip_of(f), cam=mc.group(1).strip().replace(" ", "_"),
                            box=box, x=float(mp.group(1)), y=float(mp.group(2)), answer=q.get("answer_text")))
    return out


def img(path, k=1.0):
    im = Image.open(path).convert("RGB").resize((int(round(W * k / 16)) * 16, int(round(H * k / 16)) * 16), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
    return (t - MEAN) / STD


def ground(rig, cam, u, v, zg=0.0):
    d, o = rig.ray(cam, u, v)
    if d[2] >= -1e-3: return None
    t = (zg - o[2]) / d[2]; p = o + t * d
    return p[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--img", default="img")
    ap.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    a = ap.parse_args()
    from transformers import AutoModel
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()
    nreg = int(getattr(model.config, "num_register_tokens", 0))
    Q = [q for q in questions(a.split) if os.path.isfile(f"{a.img}/{a.split}/{q['clip']}/_done.json")]
    by = {}
    for q in Q: by.setdefault((q["clip"], q["cam"]), []).append(q)
    print(f"[motion] {a.split}: {len(Q)} questions over {len(by)} clip-cameras", flush=True)
    out = {}; t0 = time.time(); meta_cache = {}
    for n, ((clip, cam), qs) in enumerate(sorted(by.items())):
        try:
            cp = f"cache/{a.split}/{clip}"
            if clip not in meta_cache:
                m = json.load(open(f"{cp}/metadata.json")); fr = m["frames"]
                e100 = load_pickle(os.path.join(cp, fr[100]["ego_state"])).pose
                e90 = load_pickle(os.path.join(cp, fr[90]["ego_state"])).pose
                dt = (fr[100]["timestamp_us"] - fr[90]["timestamp_us"]) * 1e-6
                meta_cache = {clip: (camgeo.Rig(m), fr, e100, e90, dt)}
            rig, fr, e100, e90, dt = meta_cache[clip]
            if cam not in rig.cams: continue
            with torch.inference_mode():
                def feats(path, k):
                    t = img(path, k)[None].cuda().half()
                    h_ = model(pixel_values=t).last_hidden_state[:, 1 + nreg:].float()
                    gh, gw = t.shape[2] // 16, t.shape[3] // 16
                    return F.normalize(h_, dim=-1).transpose(1, 2).reshape(-1, gh, gw)
                p100 = f"{a.img}/{a.split}/{clip}/{fr[100]['sensors']['cameras'][cam]}"
                p90 = f"{a.img}/{a.split}/{clip}/{fr[90]['sensors']['cameras'][cam]}"
                f100 = feats(p100, 1.0); f90s = {k: feats(p90, k) for k in SCALES}
            f90 = f90s[1.0]
            for q in qs:
                x0, y0, x1, y1 = q["box"]
                c0 = int(np.clip(math.floor(x0 / 1000 * GW), 0, GW - 1)); c1 = int(np.clip(math.ceil(x1 / 1000 * GW), c0 + 1, GW))
                r0 = int(np.clip(math.floor(y0 / 1000 * GH), 0, GH - 1)); r1 = int(np.clip(math.ceil(y1 / 1000 * GH), r0 + 1, GH))
                T = f100[:, r0:r1, c0:c1]                                            # [D, h, w]
                th, tw = T.shape[1:]
                sim = F.conv2d(F.pad(f90[None], (SEARCH, SEARCH, SEARCH, SEARCH)), T[None])[0, 0] / (th * tw)
                # sim[i, j]: template top-left at (i - SEARCH, j - SEARCH) in frame 90; zero shift = (r0+S, c0+S)
                i_lo, j_lo = r0, c0                                   # = (r0 + S) - S
                win = sim[i_lo: r0 + 2 * SEARCH + 1, j_lo: c0 + 2 * SEARCH + 1]
                k = int(torch.argmax(win)); di, dj = divmod(k, win.shape[1])
                dy, dx = (i_lo + di) - (r0 + SEARCH), (j_lo + dj) - (c0 + SEARCH)
                s_best = float(win.max()); s_zero = float(sim[r0 + SEARCH, c0 + SEARCH])
                # sub-patch refinement (parabola through the peak and its neighbours)
                def para(m1, m0, p1):
                    den = m1 - 2 * m0 + p1
                    return 0.0 if abs(den) < 1e-9 else float(np.clip(0.5 * (m1 - p1) / den, -0.5, 0.5))
                if 0 < di < win.shape[0] - 1: dy += para(float(win[di - 1, dj]), float(win[di, dj]), float(win[di + 1, dj]))
                if 0 < dj < win.shape[1] - 1: dx += para(float(win[di, dj - 1]), float(win[di, dj]), float(win[di, dj + 1]))
                # scale search: template centre matched in each enlarged frame-90 map
                best_k, best_s = 1.0, s_best
                sk = {}
                for kk, f9 in f90s.items():
                    if kk == 1.0: sk[kk] = s_best; continue
                    gh9, gw9 = f9.shape[1:]
                    rc, cc = int(round((r0 + r1) / 2 * kk - th / 2)), int(round((c0 + c1) / 2 * kk - tw / 2))
                    S2 = int(SEARCH * kk)
                    m9 = F.conv2d(F.pad(f9[None], (S2, S2, S2, S2)), T[None])[0, 0] / (th * tw)
                    w9 = m9[max(0, rc): rc + 2 * S2 + 1, max(0, cc): cc + 2 * S2 + 1]
                    sk[kk] = float(w9.max()) if w9.numel() else -1.0
                    if sk[kk] > best_s: best_k, best_s = kk, sk[kk]
                d100 = math.hypot(q["x"], q["y"])
                du, dv = dx * 1000 / GW, dy * 1000 / GH                             # shift of frame-90 match, 0..1000 units
                ub, vb = (x0 + x1) / 2, max(y0, y1)
                g100 = ground(rig, cam, ub, vb); g90 = ground(rig, cam, ub + du, vb + dv)
                feat = dict(du=du, dv=dv, s_best=s_best, s_zero=s_zero, dt=dt, k_best=best_k,
                            v_away_scale=(d100 - best_k * d100) / dt,
                            **{f"s_k{kk}": v for kk, v in sk.items()})
                if g100 is not None and g90 is not None:
                    # frame-90 ego frame -> world -> frame-100 ego frame
                    c9, s9 = math.cos(e90["yaw"]), math.sin(e90["yaw"])
                    wx = e90["x"] + c9 * g90[0] - s9 * g90[1]; wy = e90["y"] + s9 * g90[0] + c9 * g90[1]
                    c1_, s1_ = math.cos(e100["yaw"]), math.sin(e100["yaw"])
                    px = c1_ * (wx - e100["x"]) + s1_ * (wy - e100["y"]); py = -s1_ * (wx - e100["x"]) + c1_ * (wy - e100["y"])
                    vx, vy = (g100[0] - px) / dt, (g100[1] - py) / dt                # object velocity, frame-100 ego axes
                    r = math.hypot(q["x"], q["y"]) or 1.0
                    vr = (vx * q["x"] + vy * q["y"]) / r                              # world radial speed (+ = away)
                    feat.update(vx=vx, vy=vy, v=math.hypot(vx, vy), v_radial=vr)
                out[q["qid"]] = feat
        except Exception as e:
            print(f"  skip {clip} {cam}: {type(e).__name__}: {e}", flush=True)
        if (n + 1) % 500 == 0:
            print(f"  [{n+1}/{len(by)}] {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(f"artifacts/motion_{a.split}.json", "w"))
    print(f"[motion] {len(out)} -> artifacts/motion_{a.split}.json", flush=True)


if __name__ == "__main__":
    main()
