"""2026-11-16 validity check of the dense (extra-frame) training data. The dense LoRA's step-500 check lost 13 points on
distance (0.800 -> 0.673 on run 1's validation questions). If extra-frame questions do not match their images (frame
index, timestamp, camera), run 1 (never trained on extra frames) should score far below its validation accuracy there,
near its gray-image (no-vision) level. If the data is fine, it should score near validation.
Sample: questions at extra frames (frame != 100) of the dense pack, N per type, deterministic (crc32 of question_id).
-> <out>_s{i}of{n}.jsonl like src/vlm_infer.py (+ frame, blind). Each question alone, same Encoder as inference.
usage: python src/dense_label_check.py <model dir> <out prefix> [--adapter dir] [--n 200] [--shard i --nshards n]"""
import argparse, json, os, sys, zlib
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_data import Encoder
from vlm_data_dense import frame_images

TYPES = ["distance_from_2d_bbox", "image_region_from_3d_object", "camera_view_for_3d_object", "future_motion_label",
         "action_risk_assessment", "driving_decision_lateral"]
GRAY_TYPES = {"distance_from_2d_bbox", "image_region_from_3d_object"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("out"); ap.add_argument("--adapter", default="")
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    pool = {s: [] for s in TYPES}
    for l in open(os.environ.get("DENSE_QPACK", "artifacts/dense_q_train.jsonl")):
        r = json.loads(l)
        if r["frame"] == 100: continue
        for q in r["qs"]:
            if q.get("subcategory") in pool and q.get("choices") and "answer" in q:
                pool[q["subcategory"]].append((zlib.crc32(q["question_id"].encode()), r["clip"], r["frame"], q))
    items = []
    for s in TYPES:
        sel = sorted(pool[s], key=lambda x: x[0])
        sel = [x for x in sel if frame_images("train", x[1], x[2]) is not None][:a.n]
        items += [(c, t, q, "") for _, c, t, q in sel]
        if s in GRAY_TYPES: items += [(c, t, q, "gray") for _, c, t, q in sel]
    items = items[a.shard::a.nshards]
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    proc = AutoProcessor.from_pretrained(a.model); enc = Encoder(proc)
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).eval()
    gray = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"gray_{os.getpid()}.jpg")
    Image.new("RGB", (640, 416), (128, 128, 128)).save(gray)
    f = open(f"{a.out}_s{a.shard}of{a.nshards}.jsonl", "w")
    print(f"{len(items)} questions in shard", flush=True)
    for n, (c, t, q, blind) in enumerate(items):
        cams = frame_images("train", c, t)
        if blind == "gray": cams = [(nm, gray, gray) for nm, _, _ in cams]
        avail = [L for L in "ABCD" if L in q["choices"]]
        batch, pos = enc(cams, [q])
        with torch.no_grad():
            lg = model(**{k: v.cuda() for k, v in batch.items()}).logits[0, pos[0]].float()
        pr = torch.softmax(torch.stack([lg[enc.letter[L]] for L in avail]), 0).tolist()
        probs = dict(zip(avail, pr))
        f.write(json.dumps({"question_id": q["question_id"], "clip": c, "frame": t, "subcategory": q["subcategory"],
                            "probs": probs, "pred": max(probs, key=probs.get), "gold": q["answer"], "blind": blind}) + "\n")
        if n % 100 == 0: f.flush(); print(f"  {n}/{len(items)}", flush=True)
    f.close(); print("done", flush=True)


if __name__ == "__main__":
    main()
