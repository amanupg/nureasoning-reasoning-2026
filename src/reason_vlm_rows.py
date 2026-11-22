"""Build reasoning-VLM rows for any split from clip metadata + questions with the devkit's own
functions (submission.clips.build_temporal_context, image_selection.collect/resolve_image_slots,
build_sft.build_prompt): 8 cameras x (t-1 s, t), history first, captions, 16 images. Images come
from our cluster cache img/<split>/<clip>/ (the same files the training rows used).

usage: python src/reason_vlm_rows.py <split> <out.jsonl> [--subs a,b]
  test:        questions from cache/test/<clip>/reasoning_questions.json
  validation:  questions from vqa_val_cl docs; used as a PARITY check against
               reasoning_ws/sft_val_multiframe.jsonl (images, captions, question text must match)
"""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "recon", "devkit"))
from nureasoning.submission.clips import build_temporal_context
from nureasoning.reasoning.modules.image_selection import collect_image_slots, resolve_image_slots
from nureasoning.reasoning.build_sft import build_prompt


def rows_for_clip(split, clip, questions):
    meta = json.load(open(f"cache/{split}/{clip}/metadata.json"))
    ctx = build_temporal_context(f"img/{split}/{clip}", meta, 100, 1, 10)
    images, contexts = resolve_image_slots(
        collect_image_slots({"temporal_multiview_context": ctx}, max_images=16, num_forward_frames=2, history_first=True),
        Path("."))
    out = []
    for q in questions:
        if q.get("question_type", "choice") != "choice" or not q.get("choices"): continue
        out.append({"sample_id": f"{clip}/{q.get('question_id')}", "question_id": q.get("question_id"), "clip_dir": clip,
                    "question_type": "choice", "subcategory": q.get("subcategory"), "images": [os.path.abspath(p) for p in images],
                    "image_contexts": contexts, "question": build_prompt(q), "choices": q["choices"],
                    "gold": {"answer": q.get("answer")}})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("split"); ap.add_argument("out"); ap.add_argument("--subs", default="")
    a = ap.parse_args(); subs = set(s for s in a.subs.split(",") if s)
    n = bad = 0
    with open(a.out, "w") as f:
        for clip in sorted(os.listdir(f"cache/{a.split}")):
            if a.split == "test":
                p = f"cache/test/{clip}/reasoning_questions.json"
                if not os.path.isfile(p): continue
                qs = json.load(open(p)).get("questions") or []
            else:
                d = f"vqa_val_cl/{clip}"
                if not os.path.isdir(d): continue
                qs = [q for fn in os.listdir(d) for q in (json.load(open(os.path.join(d, fn))).get("questions") or [])]
            for r in rows_for_clip(a.split, clip, [q for q in qs if not subs or q.get("subcategory") in subs]):
                if len(r["images"]) != 16: bad += 1
                f.write(json.dumps(r) + "\n"); n += 1
    print(f"{a.split}: {n} rows -> {a.out} ({bad} rows without 16 images)")


if __name__ == "__main__":
    main()
