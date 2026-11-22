"""Score a devkit reasoning LoRA (Qwen3.5 VLM) on multiple-choice rows of a build_sft JSONL.

Inputs are built EXACTLY as nureasoning.reasoning.train.process_sample does (image captions +
images + question; processor longest_edge = max_pixels). One forward pass per question: the
sequence is the trained format with assistant "A"; the model's next-token distribution at the
position right before that letter gives P(A..D) (whatever the chat template puts between the
generation prompt and the answer, e.g. an empty think block, is identical for all letters).
-> <out>.jsonl: {sample_id, question_id, clip_dir, subcategory, gold, pred, probs}
usage: python src/reason_vlm_eval.py <adapter_dir> <sft.jsonl> <out_prefix> [--subs a,b] [--shard i --nshards n]
"""
import argparse, json, os, sys
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter"); ap.add_argument("jsonl"); ap.add_argument("out")
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B"); ap.add_argument("--max-pixels", type=int, default=200704)
    ap.add_argument("--subs", default=""); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "recon", "devkit"))
    from transformers import AutoProcessor
    from peft import PeftModel
    from nureasoning.reasoning.train import open_images, resolve_model_class, resolve_pretrained_path, from_pretrained

    subs = set(s for s in a.subs.split(",") if s)
    rows = []
    for line in open(a.jsonl):
        d = json.loads(line)
        if d.get("question_type") == "choice" and (not subs or d["subcategory"] in subs) and d.get("choices"):
            rows.append(d)
    rows = rows[a.shard::a.nshards]
    out_path = f"{a.out}_s{a.shard}of{a.nshards}.jsonl"
    done = set()
    if os.path.isfile(out_path):
        done = {json.loads(l)["sample_id"] for l in open(out_path)}
    print(f"{len(rows)} rows in shard, {len(done)} already scored", flush=True)

    mp = resolve_pretrained_path(a.base)
    proc = from_pretrained(AutoProcessor.from_pretrained, str(mp), trust_remote_code=True)
    proc.image_processor.size["longest_edge"] = int(a.max_pixels)
    model = from_pretrained(resolve_model_class(mp, None).from_pretrained, str(mp), trust_remote_code=True,
                            torch_dtype=torch.bfloat16, device_map=None).cuda().eval()
    model = PeftModel.from_pretrained(model, a.adapter).eval()
    tok = proc.tokenizer
    letter_ids = {L: tok.encode(L, add_special_tokens=False) for L in "ABCD"}
    assert all(len(v) == 1 for v in letter_ids.values()), letter_ids
    letter_ids = {L: v[0] for L, v in letter_ids.items()}

    f = open(out_path, "a")
    for n, d in enumerate(rows):
        if d["sample_id"] in done: continue
        ims = open_images(d["images"])
        ctx = d.get("image_contexts")
        content = []
        if isinstance(ctx, list) and len(ctx) == len(ims):
            for im, c in zip(ims, ctx):
                content += [{"type": "text", "text": c}, {"type": "image", "image": im}]
        else:
            content += [{"type": "image", "image": im} for im in ims]
        content.append({"type": "text", "text": d["question"]})
        full = proc.apply_chat_template([{"role": "user", "content": content},
                                         {"role": "assistant", "content": [{"type": "text", "text": "A"}]}],
                                        tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt")
        ids = full["input_ids"][0]
        pos = (ids == letter_ids["A"]).nonzero().flatten()
        p = int(pos[-1])                                   # the assistant's letter (last "A" token)
        with torch.no_grad():
            out = model(**{k: (v.cuda() if torch.is_tensor(v) else v) for k, v in full.items()})
        logits = out.logits[0, p - 1].float()
        avail = [L for L in "ABCD" if L in d["choices"]]
        probs = torch.softmax(torch.stack([logits[letter_ids[L]] for L in avail]), 0).tolist()
        pred = avail[int(max(range(len(avail)), key=lambda i: probs[i]))]
        f.write(json.dumps({"sample_id": d["sample_id"], "question_id": d["question_id"], "clip_dir": d["clip_dir"],
                            "subcategory": d["subcategory"], "gold": d["gold"]["answer"], "pred": pred,
                            "probs": dict(zip(avail, probs))}) + "\n")
        if n % 200 == 0:
            f.flush(); print(f"  {n}/{len(rows)}", flush=True)
    f.close(); print("done", flush=True)


if __name__ == "__main__":
    main()
