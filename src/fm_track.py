"""Track the box a future_motion_label question states, backwards from frame 100, with a SAM video
tracker (SAM 3 tracker or SAM 2.1, transformers). Raw measurement only; geometry + rule in fm_rule.py.

For each clip and camera named by its future-motion questions: frames 90..100 of that camera found in
the image roots (first root that has the file wins), reversed so frame 100 is index 0, every question's
stated box (0..1000) as one object prompt at index 0, then propagation. Per object and frame: mask
bounding box (0..1000), mask area (fraction of the image), object score logit, camera timestamp (s).

-> <out>_s{i}of{n}.json: {question_id: {clip, cam, box_q, x, y, answer, frames, t, box, area, score}}
usage: python src/fm_track.py <split> <out_prefix> [--roots img_fm:img_a15:img] [--model facebook/sam3]
       [--sub future_motion_label | distance_from_2d_bbox]
       [--shard i --nshards n]
"""
import argparse, json, os, re, sys
import numpy as np
import torch
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from motion_corr import questions
RE_TS = re.compile(r"_(\d+)\.jpg$")


def frame_path(roots, split, clip, rel):
    for r in roots:
        p = f"{r}/{split}/{clip}/{rel}"
        if os.path.isfile(p): return p
    return None


def load_model(name):
    import transformers as T
    if "sam3" in name:
        M, P = T.Sam3TrackerVideoModel, T.Sam3TrackerVideoProcessor
    else:
        M, P = T.Sam2VideoModel, T.Sam2VideoProcessor
    return M.from_pretrained(name).cuda().to(torch.bfloat16).eval(), P.from_pretrained(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out")
    ap.add_argument("--roots", default=f"img_fm:img_a15:{os.environ.get('IMG_ROOT', 'img')}"); ap.add_argument("--model", default="facebook/sam3")
    ap.add_argument("--first", type=int, default=90); ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--sub", default="future_motion_label")
    a = ap.parse_args(); roots = a.roots.split(":")
    if a.sub == "distance_from_2d_bbox":
        import dist_mono as dist_geo
        qs = [dict(qid=x["q"]["question_id"], clip=x["clip"], cam=x["cam"], box=x["box"], x=None, y=None,
                   answer=x["q"].get("answer"), cat=x["cat"], choices=x["q"]["choices"]) for x in dist_geo.questions(a.split)]
    else:
        qs = questions(a.split)
    groups = {}
    for q in qs: groups.setdefault((q["clip"], q["cam"]), []).append(q)
    keys = sorted(groups)[a.shard::a.nshards]
    out_path = f"{a.out}_s{a.shard}of{a.nshards}.json"
    res = json.load(open(out_path)) if os.path.isfile(out_path) else {}
    print(f"{len(qs)} questions, {len(keys)} clip-cameras in shard, {len(res)} done", flush=True)
    model, proc = load_model(a.model)
    for n, (clip, cam) in enumerate(keys):
        G = [q for q in groups[(clip, cam)] if q["qid"] not in res]
        if not G: continue
        fr = json.load(open(f"cache/{a.split}/{clip}/metadata.json"))["frames"]
        ks, paths = [], []
        for k in range(100, a.first - 1, -1):
            rel = fr[k]["sensors"]["cameras"].get(cam)
            p = frame_path(roots, a.split, clip, rel) if rel else None
            if p: ks.append(k); paths.append(p)
        if not ks or ks[0] != 100:
            for q in G: res[q["qid"]] = dict(clip=clip, cam=cam, missing=True)
            continue
        ims = [Image.open(p).convert("RGB") for p in paths]
        W, H = ims[0].size
        ims = [im if im.size == (W, H) else im.resize((W, H), Image.BILINEAR) for im in ims]
        sess = proc.init_video_session(video=ims, inference_device="cuda", dtype=torch.bfloat16)
        ids = list(range(1, len(G) + 1))
        boxes = [[[q["box"][0] * W / 1000, q["box"][1] * H / 1000, q["box"][2] * W / 1000, q["box"][3] * H / 1000] for q in G]]
        proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=0, obj_ids=ids, input_boxes=boxes)
        rec = {q["qid"]: dict(clip=clip, cam=cam, box_q=q["box"], x=q["x"], y=q["y"], answer=q["answer"],
                              cat=q.get("cat"), choices=q.get("choices"),
                              frames=ks, t=[int(RE_TS.search(p).group(1)) / 1e6 for p in paths],
                              box=[None] * len(ks), area=[0.0] * len(ks), score=[None] * len(ks)) for q in G}
        with torch.inference_mode():
            for o in model.propagate_in_video_iterator(sess, start_frame_idx=0):
                m = proc.post_process_masks([o.pred_masks], original_sizes=[[H, W]], binarize=True)[0]
                sc = o.object_score_logits.float().flatten().tolist() if o.object_score_logits is not None else [None] * len(G)
                for j, q in enumerate(G):
                    mk = m[j, 0].cpu().numpy() if m.dim() == 4 else m[j].cpu().numpy()
                    r = rec[q["qid"]]; i = o.frame_idx
                    r["score"][i] = sc[j]
                    if mk.any():
                        ys, xs = np.nonzero(mk)
                        r["box"][i] = [1000 * xs.min() / W, 1000 * ys.min() / H, 1000 * (xs.max() + 1) / W, 1000 * (ys.max() + 1) / H]
                        r["area"][i] = float(mk.mean())
        res.update(rec)
        if n % 20 == 0:
            json.dump(res, open(out_path, "w")); print(f"  {n}/{len(keys)}", flush=True)
    json.dump(res, open(out_path, "w")); print("done", len(res), flush=True)


if __name__ == "__main__":
    main()
