"""Letter probabilities from a (LoRA-tuned) Qwen3.5-family VLM, one question per forward (src/vlm_data.py
format, k = 1). -> <out>_s{i}of{n}.jsonl: {question_id, clip, subcategory, probs, pred, gold}
usage: python src/vlm_infer.py <split> <model dir> <out prefix> [--adapter dir] [--shard i --nshards n] [--limit n]
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_data import Encoder, clip_images, load_questions


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("split"); ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--adapter", default=""); ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--subs", default=""); ap.add_argument("--cap", type=int, default=0,
                    help="at most this many questions per subcategory (deterministic: smallest question ids)")
    ap.add_argument("--qids", default="", help="file with one question_id per line (only these)")
    ap.add_argument("--perm", type=int, default=0, help="cyclic shift k of the option contents over the letters; "
                    "probabilities are written back under the ORIGINAL letters (for order-bias averaging)")
    ap.add_argument("--blind", default="", choices=["", "gray", "swap"],
                    help="text-reliance diagnostic: 'gray' = every image replaced by mid-gray; 'swap' = images of "
                         "another clip of the same split (sorted clip list shifted by 97)")
    a = ap.parse_args()
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(a.model); enc = Encoder(proc)
    if torch.cuda.device_count() > 1:            # big base (e.g. Cosmos3-Super 32B tower): split layers over the visible GPUs
        model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="balanced").eval()
    else:
        model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).eval()
    subs = set(s for s in a.subs.split(",") if s)
    Q = load_questions(a.split)
    keep = set(open(a.qids).read().split()) if a.qids else None
    items = [(c, q) for c in sorted(Q) for q in Q[c] if (not subs or q.get("subcategory") in subs)
             and (keep is None or q["question_id"] in keep)]
    if a.cap:
        by = {}
        for c, q in sorted(items, key=lambda x: x[1]["question_id"]): by.setdefault(q.get("subcategory"), []).append((c, q))
        items = sorted([x for v in by.values() for x in v[:a.cap]], key=lambda x: (x[0], x[1]["question_id"]))
    items = items[a.shard::a.nshards]
    if a.limit: items = items[:a.limit]
    path = f"{a.out}_s{a.shard}of{a.nshards}.jsonl"
    done = set()
    if os.path.isfile(path):                    # tolerate a truncated last line (preempted / requeued job)
        good = []
        for l in open(path):
            try: done.add(json.loads(l)["question_id"]); good.append(l)
            except Exception: pass
        open(path, "w").writelines(good)
    print(f"{len(items)} questions in shard, {len(done)} done", flush=True)
    f = open(path, "a"); cams_cache = {}
    clips_sorted = sorted(Q); swap_of = {c: clips_sorted[(i + 97) % len(clips_sorted)] for i, c in enumerate(clips_sorted)}
    gray = None
    if a.blind == "gray":
        from PIL import Image
        gray = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"gray_{os.getpid()}.jpg")
        Image.new("RGB", (640, 416), (128, 128, 128)).save(gray)
    for n, (c, q) in enumerate(items):
        if q["question_id"] in done: continue
        if c not in cams_cache:
            cams_cache = {c: clip_images(a.split, c)}
            if cams_cache[c] is not None and a.blind == "swap":
                cs = c
                for _ in range(len(clips_sorted)):          # next clip in the shifted order that has images
                    cs = swap_of[cs]; alt = clip_images(a.split, cs)
                    if alt is not None and cs != c: break
                cams_cache[c] = alt
            elif cams_cache[c] is not None and a.blind == "gray":
                cams_cache[c] = [(n_, gray, gray) for n_, _, _ in cams_cache[c]]
        cams = cams_cache[c]
        if cams is None: continue
        avail = [L for L in "ABCD" if L in q["choices"]]
        if a.perm:      # letter avail[i] now shows the content of avail[(i + k) % n]
            n_ = len(avail); q = dict(q, choices={avail[i]: q["choices"][avail[(i + a.perm) % n_]] for i in range(n_)})
        batch, pos = enc(cams, [q])
        with torch.no_grad():
            lg = model(**{k: v.cuda() for k, v in batch.items()}).logits[0, pos[0]].float()
        pr = torch.softmax(torch.stack([lg[enc.letter[L]] for L in avail]), 0).tolist()
        probs = {avail[(i + a.perm) % len(avail)]: pr[i] for i in range(len(avail))}       # back to original letters
        f.write(json.dumps({"question_id": q["question_id"], "clip": c, "subcategory": q.get("subcategory"),
                            "probs": probs, "pred": max(probs, key=probs.get), "gold": q.get("answer"), "perm": a.perm,
                            "blind": a.blind}) + "\n")
        if n % 200 == 0: f.flush(); print(f"  {n}/{len(items)}", flush=True)
    f.close(); print("done", flush=True)


if __name__ == "__main__":
    main()
