"""Is the object a question box points at actually visible in the key frame? (inputs only: question text +
images). SAM 3 text-prompted detection of the stated category on the stated camera's key frame; a box counts
as visible when a detection of that category overlaps it (IoU >= 0.3 or covers >= 50% of the box).
-> <out>.jsonl {question_id, split, sub, cat, cam, box, best_iou, best_cover, best_score, n_det}
usage: python src/box_visible.py <split> <out.jsonl> [--subs a,b] [--max 2000]"""
import argparse, json, os, random, re, sys
import torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camgeo
from vlm_data import load_questions, clip_images
R = re.compile(r"the ([\w ]+?) (?:object )?in the (.+?) camera with normalized 2D box \[(\d+), (\d+), (\d+), (\d+)\]")
PROMPT = {"car": "car", "human": "person", "traffic_cone": "traffic cone", "bicycle": "bicycle"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("split"); ap.add_argument("out")
    ap.add_argument("--subs", default="distance_from_2d_bbox,future_motion_label,ego_position_from_2d_bbox")
    ap.add_argument("--max", type=int, default=2000); a = ap.parse_args()
    from transformers import Sam3Model, Sam3Processor
    proc = Sam3Processor.from_pretrained("facebook/sam3"); model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.bfloat16).cuda().eval()
    subs = set(a.subs.split(",")); items = []
    for clip, qs in load_questions(a.split).items():
        for q in qs:
            if q["subcategory"] not in subs: continue
            m = R.search(q["question"])
            if m and m.group(1) in PROMPT: items.append((clip, q, m))
    random.Random(0).shuffle(items); items = items[:a.max]
    done = set()
    if os.path.isfile(a.out): done = {json.loads(l)["question_id"] for l in open(a.out)}
    fo = open(a.out, "a"); cache = {}
    for i, (clip, q, m) in enumerate(items):
        if q["question_id"] in done: continue
        cat, cam = m.group(1), m.group(2).replace(" ", "_"); box = [int(v) for v in m.groups()[2:6]]
        key = camgeo.NAME2KEY.get(cam); ims = clip_images(a.split, clip)
        p = [k for _, _, k in (ims or []) if key and f"/{key}/" in k]
        if not p: continue
        ck = (p[0], cat)
        if ck not in cache:
            im = Image.open(p[0]).convert("RGB"); W, H = im.size
            inp = proc(images=im, text=PROMPT[cat], return_tensors="pt").to("cuda")
            inp = {k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v) for k, v in inp.items()}
            with torch.no_grad(): out = model(**inp)
            res = proc.post_process_instance_segmentation(out, threshold=0.3, mask_threshold=0.5, target_sizes=[(H, W)])[0]
            dets = [([float(b[0]) / W * 1000, float(b[1]) / H * 1000, float(b[2]) / W * 1000, float(b[3]) / H * 1000], float(s))
                    for b, s in zip(res["boxes"], res["scores"])]
            cache = {ck: dets} if len(cache) > 64 else {**cache, ck: dets}
        dets = cache[ck]; bi = bc = bs = 0.0
        ba = max((box[2] - box[0]) * (box[3] - box[1]), 1)
        for d, s in dets:
            ix = max(0, min(box[2], d[2]) - max(box[0], d[0])) * max(0, min(box[3], d[3]) - max(box[1], d[1]))
            da = max((d[2] - d[0]) * (d[3] - d[1]), 1); iou = ix / (ba + da - ix); cov = ix / ba
            if iou > bi: bi, bs = iou, s
            bc = max(bc, cov)
        fo.write(json.dumps({"question_id": q["question_id"], "split": a.split, "sub": q["subcategory"], "cat": cat, "cam": cam,
                             "box": box, "best_iou": round(bi, 3), "best_cover": round(bc, 3), "best_score": round(bs, 3), "n_det": len(dets)}) + "\n")
        if i % 100 == 0: fo.flush(); print(i, len(items), flush=True)
    fo.close()


if __name__ == "__main__":
    main()
