"""Alpamayo2-Edge (nvidia/Alpamayo2-Edge, released 2026-11-10) zero-shot on nuReasoning multiple-choice questions.
Inputs follow the model's VQA profile (4 frames at t0-0.3..t0 s, 163840..196608 pixels per image, camera name + relative
time label before each image, no navigation/ego history), built here from our cameras:
  --cams native6 : the Edge rig names (front left / front / front right / rear left / rear right / front telephoto =
                   2x centre crop of our front camera); our left, right and back cameras are not shown.
  --cams all8    : all 8 of our cameras with plain names (Left camera, Rear camera, ...).
Question + lettered options + "Answer with the letter...", then the template's closed <think></think>; the answer is
the option letter with the highest first-token logit (the greedy 4-token text is logged as a check).
-> <out>_s<i>of<n>.jsonl, one record per question {question_id, clip, subcategory, probs, pred, gold, gen}.
usage: python src/edge_vqa.py <split> <model dir> <out prefix> [--cams all8] [--subs a,b] [--cap-per-type n] [--shard i --nshards n]
"""
import argparse, glob, json, os, random, sys, time
from collections import Counter
import torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edge_members import frames

TEST_TYPES = ("image_region_from_3d_object future_motion_label distance_from_2d_bbox action_risk_assessment "
              "camera_view_for_3d_object driving_decision_joint driving_decision_lateral multiview_camera_pair "
              "driving_decision_longitudinal ego_position_from_2d_bbox unsafe_action_identification object_relative_position "
              "future_path_intersection closest_object_type conflict_prediction motion_relation").split()
NATIVE6 = [("Front left camera", "front_left"), ("Front camera", "front"), ("Front right camera", "front_right"),
           ("Rear left camera", "back_left"), ("Rear right camera", "back_right"), ("Front telephoto camera", "TELE")]
ALL8 = [("Front left camera", "front_left"), ("Front camera", "front"), ("Front right camera", "front_right"),
        ("Left camera", "left"), ("Right camera", "right"), ("Rear left camera", "back_left"), ("Rear camera", "back"),
        ("Rear right camera", "back_right")]
INSTR = "Answer with the letter of the correct option only."
PIX = (163840, 196608)


def load_questions(split):
    Q = {}
    if split == "validation":
        for f in sorted(glob.glob("vqa_val/*/*_vqa.json")):
            c = f.split("/")[1]
            Q.setdefault(c, []).extend(q for q in json.load(open(f))["questions"] if q.get("question_type") == "choice")
    else:
        for c in sorted(os.listdir("cache/test")):
            p = f"cache/test/{c}/reasoning_questions.json"
            if os.path.isfile(p): Q[c] = json.load(open(p))["questions"]
    return Q


def image_content(split, clip, img_root, cams):
    md = json.load(open(f"cache/{split}/{clip}/metadata.json")); fs = frames(md)
    t_last = fs[-1]["timestamp_us"]; content = []
    for name, cam in cams:
        content.append({"type": "text", "text": name + ": "})
        for f in fs:
            src = "front" if cam == "TELE" else cam
            im = Image.open(os.path.join(img_root, split, clip, f["sensors"]["cameras"][src])).convert("RGB")
            if cam == "TELE":
                W, H = im.size; im = im.crop((W // 4, H // 4, W - W // 4, H - H // 4))
            content.append({"type": "text", "text": f"<t={(f['timestamp_us'] - t_last) / 1e6:z.1f}s>"})
            x = torch.from_numpy(__import__("numpy").asarray(im)).permute(2, 0, 1).unsqueeze(0)   # [1, 3, H, W] uint8
            content.append({"type": "image", "image": x, "min_pixels": PIX[0], "max_pixels": PIX[1]})
    return content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--cams", default="native6", choices=["native6", "all8"])
    ap.add_argument("--subs", default=",".join(TEST_TYPES)); ap.add_argument("--cap-per-type", type=int, default=0)
    ap.add_argument("--clips", default="", help="file with clip names (default: every clip with images)")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    from alpamayo2_edge.checkpoint import load_model
    from alpamayo2_edge.helper import _tokenize_messages, get_processor, to_device
    from alpamayo2_edge.chat_template import DEFAULT_SYSTEM_PROMPT
    img_root = os.environ["IMG_ROOT"]; cams = NATIVE6 if a.cams == "native6" else ALL8
    subs = set(a.subs.split(","))
    Q = load_questions(a.split)
    have = {c for c in Q if os.path.isfile(os.path.join(img_root, a.split, c, "_done.json"))}
    if a.clips: have &= set(open(a.clips).read().split())
    items = [(c, q) for c in sorted(have) for q in Q[c] if q["subcategory"] in subs and q.get("choices")]
    if a.cap_per_type:
        random.Random(0).shuffle(items); n = Counter(); keep = []
        for c, q in items:
            if n[q["subcategory"]] < a.cap_per_type: keep.append((c, q)); n[q["subcategory"]] += 1
        items = sorted(keep, key=lambda x: x[0])
    items = items[a.shard::a.nshards]
    out = f"{a.out}_s{a.shard}of{a.nshards}.jsonl"
    done = {json.loads(l)["question_id"] for l in open(out)} if os.path.isfile(out) else set()
    print(f"[edge] {a.split} {a.cams}: {len(have)} clips with images, {len(items)} questions ({len(done)} done), "
          f"types {dict(Counter(q['subcategory'] for _, q in items))}", flush=True)
    model, _ = load_model(a.model, device="cuda", dtype="bfloat16", attn_implementation="sdpa")
    model.eval(); proc = get_processor(model.tokenizer, model.config); tok = model.tokenizer
    letter_id = {}
    for L in "ABCDEFGH":
        ids = tok.encode(L, add_special_tokens=False); assert len(ids) == 1, (L, ids); letter_id[L] = ids[0]
    system = getattr(model.config, "system_prompt", None) or DEFAULT_SYSTEM_PROMPT
    fh = open(out, "a"); t0 = time.time(); cache = (None, None); n = 0; ok = 0
    for c, q in items:
        if q["question_id"] in done: continue
        if cache[0] != c: cache = (c, image_content(a.split, c, img_root, cams))
        opts = "\n".join(f"{k}. {v}" for k, v in sorted(q["choices"].items()))
        content = cache[1] + [{"type": "text", "text": "\n"},
                              {"type": "text", "text": f"{q['question'].strip()}\nOptions:\n{opts}\n{INSTR}"}]
        msgs = [{"role": "system", "content": [{"type": "text", "text": system}]}, {"role": "user", "content": content}]
        td = to_device(_tokenize_messages(msgs, proc, include_closed_thinking=True), device="cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            g = model.vlm.generate(**td, max_new_tokens=4, do_sample=False, num_beams=1, output_logits=True,
                                   return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
        lg = g.logits[0][0].float()
        Ls = sorted(q["choices"]); z = torch.stack([lg[letter_id[L]] for L in Ls]); p = torch.softmax(z, 0).tolist()
        pred = Ls[max(range(len(Ls)), key=lambda i: p[i])]
        gen = tok.decode(g.sequences[0, td["input_ids"].shape[1]:], skip_special_tokens=False)
        rec = {"question_id": q["question_id"], "clip": c, "subcategory": q["subcategory"], "probs": dict(zip(Ls, p)),
               "pred": pred, "gold": q.get("answer"), "gen": gen, "ntok": int(td["input_ids"].shape[1])}
        fh.write(json.dumps(rec) + "\n"); fh.flush(); n += 1; ok += rec["gold"] == pred
        if n <= 5 or n % 100 == 0:
            print(f"  {n} {time.time() - t0:.0f}s acc {ok / n:.3f} ntok {rec['ntok']} gen {gen!r}", flush=True)
    print(f"[edge] done {n} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
