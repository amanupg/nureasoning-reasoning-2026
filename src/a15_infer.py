"""Alpamayo 1.5 (nvidia/Alpamayo-1.5-10B) trajectories for nuReasoning clips.

Inputs mapped from the Motional rig to Alpamayo's camera slots:
  CAM_M_L0 -> 0 "Front left", CAM_M_F -> 1 "Front", CAM_M_R0 -> 2 "Front right"
  (no telephoto camera; Alpamayo supports variable camera counts).
4 frames per camera at 10 Hz ending at the key frame (t0), 16-step ego history
(15 history points + current pose) in the t0 ego frame, z = 0 (history has no z).
Output per clip: all sampled local trajectories (S, 64, 3) at t0+0.1..6.4 s, the
rotations, and the chain-of-causation text. Resumable: skips clips already done.
"""
import argparse, glob, json, math, os, sys, time
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "recon", "devkit")]
from nureasoning.common.pickle_io import load_pickle          # noqa: E402
from alpamayo1_5 import helper                                  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5          # noqa: E402

CAMS = [("front_left", 0), ("front", 1), ("front_right", 2)]
FRAMES = [97, 98, 99, 100]
KI = 100


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def route_nav_text(frame, ego):
    """'Turn left in 30m' style text from the route command, as in Alpamayo's nav data.
    Distance = arc length along the route polyline to the first point whose heading
    departs >30 deg from the route's initial heading. None for non-turn commands."""
    mg = frame.get("mission_goal") or {}
    cmd = (mg.get("command") or "").upper()
    word = {"TURN_LEFT": "left", "TURN_RIGHT": "right"}.get(cmd)
    if word is None:
        return None
    pts = np.array(mg.get("route_path") or [], float)
    x0, y0, yaw = ego.pose["x"], ego.pose["y"], ego.pose["yaw"]
    if len(pts) < 2:
        return f"Turn {word}"
    loc = (pts - [x0, y0]) @ rotz(yaw)[:2, :2]          # route in ego frame
    seg = np.diff(loc, axis=0); L = np.linalg.norm(seg, axis=1)
    hd = np.arctan2(seg[:, 1], seg[:, 0])
    s = np.concatenate([[0.0], np.cumsum(L)])
    start = int(np.argmin(np.linalg.norm(loc, axis=1)))
    h0 = hd[min(start, len(hd) - 1)]
    for i in range(start, len(hd)):
        if abs((hd[i] - h0 + np.pi) % (2 * np.pi) - np.pi) > math.radians(30):
            d = max(0.0, s[i] - s[start])
            return f"Turn {word} in {int(round(d))}m"
    return f"Turn {word}"


def load_inputs(cache_clip, img_clip):
    meta = json.load(open(os.path.join(cache_clip, "metadata.json")))
    fr = meta["frames"]
    ego = load_pickle(os.path.join(cache_clip, fr[KI]["ego_state"]))
    imgs = []
    for cam, _ in CAMS:
        for i in FRAMES:
            p = os.path.join(img_clip, fr[i]["sensors"]["cameras"][cam])
            imgs.append(torch.from_numpy(np.asarray(Image.open(p).convert("RGB"))).permute(2, 0, 1))
    x0, y0, yaw0 = ego.pose["x"], ego.pose["y"], ego.pose["yaw"]
    h = np.array(ego.trajectory_history, float)[-15:]
    h = np.vstack([h, [x0, y0, yaw0]])                   # 16 steps ending at t0
    R0 = rotz(yaw0)
    xy = (h[:, :2] - [x0, y0]) @ R0[:2, :2]              # world -> t0 ego frame
    hx = np.column_stack([xy, np.zeros(len(h))])
    hr = np.stack([rotz(a - yaw0) for a in h[:, 2]])
    return {"frames": torch.stack(imgs), "cam_idx": torch.tensor([c for _, c in CAMS]),
            "hist_xyz": torch.from_numpy(hx).float()[None, None],
            "hist_rot": torch.from_numpy(hr).float()[None, None],
            "nav": route_nav_text(fr[KI], ego), "pose": (x0, y0, yaw0)}


def load_model(repo, vlm):
    """The checkpoint's config names nvidia/Cosmos-Reason2-8B (gated) only as the source
    of the Qwen3-VL architecture config and tokenizer; every weight comes from the
    Alpamayo checkpoint. Substituting the public Qwen3-VL-8B config is accepted ONLY if
    it reproduces the stored vocab size and trajectory-token offset and all weights
    load with zero missing/unexpected keys."""
    from huggingface_hub import snapshot_download
    snap = snapshot_download(repo)
    d = json.load(open(os.path.join(snap, "config.json")))
    want = (d["vocab_size"], d["traj_token_start_idx"])
    d["vlm_name_or_path"] = snapshot_download(vlm) if not os.path.isdir(vlm) else vlm
    d["attn_implementation"] = "sdpa"
    cfg = Alpamayo1_5.config_class(**d)
    got = (cfg.vocab_size, cfg.traj_token_start_idx)
    assert got == want, f"tokenizer mismatch: {got} != {want}"
    model, info = Alpamayo1_5.from_pretrained(snap, config=cfg, dtype=torch.bfloat16,
                                              output_loading_info=True)
    bad = {k: v for k, v in info.items() if v}
    assert not bad, f"loading info not clean: { {k: list(v)[:5] for k, v in bad.items()} }"
    print(f"[a15] model ok: vocab {got}, all weights loaded", flush=True)
    return model.to("cuda").eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--img", default="img_a15")
    ap.add_argument("--out", default="a15_out")
    ap.add_argument("--variant", default="nonav", choices=["nonav", "nav"])
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--model", default="nvidia/Alpamayo-1.5-10B")
    ap.add_argument("--vlm", default="Qwen/Qwen3-VL-8B-Instruct",
                    help="config/processor source replacing the gated nvidia/Cosmos-Reason2-8B")
    a = ap.parse_args()

    clips = sorted(os.path.basename(os.path.dirname(d))
                   for d in glob.glob(f"{a.img}/{a.split}/*/_done.json"))[a.shard::a.nshards]
    if a.limit: clips = clips[:a.limit]
    out = os.path.join(a.out, a.variant, a.split); os.makedirs(out, exist_ok=True)
    todo = [c for c in clips if not os.path.isfile(os.path.join(out, c + ".npz"))]
    print(f"[a15] {a.variant} {a.split}: {len(todo)}/{len(clips)} to do", flush=True)
    if not todo: return

    model = load_model(a.model, a.vlm)
    proc = helper.get_processor(model.tokenizer)
    t0 = time.time()
    for n, c in enumerate(todo):
        try:
            d = load_inputs(os.path.join("cache", a.split, c), os.path.join(a.img, a.split, c))
        except Exception as e:
            print(f"  skip {c}: {type(e).__name__}: {e}", flush=True); continue
        nav = d["nav"] if a.variant == "nav" else None
        msgs = helper.create_message(d["frames"], camera_indices=d["cam_idx"], nav_text=nav)
        tok = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                                       continue_final_message=True, return_dict=True,
                                       return_tensors="pt")
        inp = helper.to_device({"tokenized_data": tok, "ego_history_xyz": d["hist_xyz"],
                                "ego_history_rot": d["hist_rot"]}, "cuda")
        torch.cuda.manual_seed_all(42)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            xyz, rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=inp, top_p=0.98, temperature=0.6, num_traj_samples=a.samples,
                max_generation_length=256, return_extra=True)
        cot = extra["cot"][0]
        cot = [str(x) for x in np.array(cot, dtype=object).ravel()] if not isinstance(cot, str) else [cot]
        np.savez_compressed(os.path.join(out, c + ".npz"),
                            xyz=xyz[0, 0].float().cpu().numpy(), rot=rot[0, 0].float().cpu().numpy(),
                            pose=np.array(d["pose"]), nav=np.array(nav or ""), cot=np.array(cot))
        if n < 3 or (n + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  [{n+1}/{len(todo)}] {el/(n+1):.1f}s/clip  end={xyz[0,0,0,-1,:2].tolist()}  "
                  f"nav={nav!r}  cot={cot[0][:100]!r}", flush=True)
    print(f"[a15] done {len(todo)} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
