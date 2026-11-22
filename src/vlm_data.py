"""Inputs for the multi-question VLM (Qwen3.5 family, e.g. the Qwen-Drive-1.0 VLM).

One example = one clip: 8 cameras, each with a view tag and two frames (about 1 s apart: the older one
small, the key frame larger), then 1-3 multiple-choice questions about that clip. The assistant answers
with one option letter per line. Only the letters are trained (labels elsewhere = -100).

Questions: train / validation from the devkit generator files (vqa_train_cl, vqa_val_cl); test from
cache/test/<clip>/reasoning_questions.json. Images: img/<split>/<clip>/cameras/<CAM>/ (two files each).
"""
import glob, json, os, random, re
from PIL import Image

CAMS = [("front", "CAM_M_F"), ("front_left", "CAM_M_L0"), ("front_right", "CAM_M_R0"), ("left", "CAM_M_L1"),
        ("right", "CAM_M_R1"), ("back_left", "CAM_M_L2"), ("back_right", "CAM_M_R2"), ("back", "CAM_M_B")]
TAG = {n: f"<{n.replace('_', ' ').upper()} VIEW>" for n, _ in CAMS}
CUR_WH, HIST_WH = (640, 416), (352, 224)          # multiples of 32 (patch 16 x merge 2)
INSTR = "Answer each question with the letter of the correct option only, one letter per line, in order."
# test question mix (cache/test reasoning_questions.json, 3,000 questions)
TEST_SHARE = {"image_region_from_3d_object": .133, "future_motion_label": .133, "distance_from_2d_bbox": .133,
              "action_risk_assessment": .126, "camera_view_for_3d_object": .089, "driving_decision_joint": .063,
              "driving_decision_lateral": .057, "multiview_camera_pair": .054, "driving_decision_longitudinal": .046,
              "ego_position_from_2d_bbox": .042, "unsafe_action_identification": .040, "object_relative_position": .029,
              "future_path_intersection": .020, "closest_object_type": .014, "conflict_prediction": .011,
              "motion_relation": .007}


def load_questions(split):
    """{clip: [question dict]} with only multiple-choice questions."""
    out = {}
    if split == "test":
        for f in glob.glob("cache/test/*/reasoning_questions.json"):
            qs = json.load(open(f)).get("questions") or []
            out[f.split("/")[2]] = qs
    else:
        root = "vqa_train_cl" if split == "train" else "vqa_val_cl"
        for f in glob.glob(f"{root}/*/*_vqa.json"):
            out.setdefault(f.split("/")[1], []).extend(json.load(open(f)).get("questions") or [])
    if os.environ.get("META_Q") == "1":          # ego meta-action questions (src/meta_q.py), one per clip
        mp = f"artifacts/meta_q_{split}.json"
        for c, qs in json.load(open(mp)).items():
            if c in out or split == "test" or os.environ.get("META_ALL_CLIPS") == "1": out.setdefault(c, []).extend(qs)
    return {c: [q for q in qs if q.get("question_type", "choice") == "choice" and q.get("choices")]
            for c, qs in out.items()}


def clip_images(split, clip, img_root=None):
    """[(cam name, older path, key-frame path)] or None if any camera lacks two frames."""
    img_root = img_root or os.environ.get("IMG_ROOT", "img")      # IMG_ROOT: frames staged per job (cluster/stage_img.sh)
    res = []
    for name, key in CAMS:
        fs = sorted(glob.glob(f"{img_root}/{split}/{clip}/cameras/{key}/*.jpg"),
                    key=lambda p: int(os.path.basename(p).rsplit("_", 1)[1].split(".")[0]))
        if len(fs) < 2: return None
        res.append((name, fs[-2], fs[-1]))
    return res


def question_text(i, q, multi):
    opts = "\n".join(f"{k}. {v}" for k, v in sorted(q["choices"].items()))
    return (f"Q{i + 1}: " if multi else "") + q["question"].strip() + "\n" + opts


# 2026-11-14 optional inputs (env-gated; default off = every earlier run):
#   FOCUS=1: the camera(s) named in the question text again as a high-resolution key frame (FOCUS_WH, at most 2);
#            test shift is toward far / small / side-camera objects, where the 640x416 key frame loses detail.
#   EGO=1:   one text line of ego kinematics at the key frame from artifacts/clipfeat_<split>.json (src/clipfeat.py,
#            same computation for train / validation / test).
FOCUS_WH = (1088, 704)                               # 766K px, multiples of 32; native frames are 2816x1856
_CAM_RE = re.compile(r"\bin the (front left|front right|back left|back right|front|back|left|right) camera\b")
_CLIPFEAT = {}


def focus_cams(qs):
    out = []
    for q in qs:
        m = _CAM_RE.search(q["question"])
        if m and m.group(1).replace(" ", "_") not in out: out.append(m.group(1).replace(" ", "_"))
    return out[:2]


def ego_line(cur_path):
    """cur_path = <img_root>/<split>/<clip>/cameras/<CAM>/<file>.jpg -> 'Ego state: ...' or ''."""
    parts = cur_path.split(os.sep); split, clip = parts[-5], parts[-4]
    if split not in _CLIPFEAT:
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", f"clipfeat_{split}.json")
        _CLIPFEAT[split] = json.load(open(p)) if os.path.isfile(p) else {}
    f = _CLIPFEAT[split].get(clip)
    if not f or "c_v" not in f: return ""
    s = (f"Ego state at frame 1: speed {f['c_v']:.1f} m/s, longitudinal acceleration {f['c_ax']:+.2f} m/s^2, "
         f"lateral acceleration {f['c_ay']:+.2f} m/s^2")
    if "c_omega" in f: s += f", yaw rate {f['c_omega']:+.3f} rad/s, speed trend over the past track {f['c_dv']:+.1f} m/s"
    return s + "."


def messages(cams, qs):
    content = []
    for name, older, cur in cams:
        content.append({"type": "text", "text": TAG[name] + "\nframe: 0"})
        content.append({"type": "image", "image": Image.open(older).convert("RGB").resize(HIST_WH, Image.BILINEAR)})
        content.append({"type": "text", "text": "frame: 1"})
        content.append({"type": "image", "image": Image.open(cur).convert("RGB").resize(CUR_WH, Image.BILINEAR)})
    if os.environ.get("FOCUS") == "1":
        key = {n: cur for n, _, cur in cams}
        for n in focus_cams(qs):
            if n in key:
                content.append({"type": "text", "text": f"<{n.replace('_', ' ').upper()} VIEW, HIGH RESOLUTION>\nframe: 1"})
                content.append({"type": "image", "image": Image.open(key[n]).convert("RGB").resize(FOCUS_WH, Image.BILINEAR)})
    if os.environ.get("EGO") == "1" and cams:
        e = ego_line(cams[0][2])
        if e: content.append({"type": "text", "text": e})
    multi = len(qs) > 1
    content.append({"type": "text", "text": "\n\n".join(question_text(i, q, multi) for i, q in enumerate(qs)) + "\n\n" + INSTR})
    return [{"role": "user", "content": content}]


class Encoder:
    """Prompt via the processor's chat template (no thinking), then the answer letters appended."""

    def __init__(self, processor):
        self.p = processor; tok = processor.tokenizer
        ip = processor.image_processor
        MAXPIX = (FOCUS_WH if os.environ.get("FOCUS") == "1" else CUR_WH)[0] * (FOCUS_WH if os.environ.get("FOCUS") == "1" else CUR_WH)[1]
        if hasattr(ip, "size") and isinstance(ip.size, dict):
            ip.size = {"shortest_edge": 32 * 32, "longest_edge": MAXPIX}
        for k, v in (("min_pixels", 32 * 32), ("max_pixels", MAXPIX)):
            if hasattr(ip, k): setattr(ip, k, v)
        self.letter = {L: tok.encode(L, add_special_tokens=False) for L in "ABCD"}
        assert all(len(v) == 1 for v in self.letter.values()), self.letter
        self.letter = {L: v[0] for L, v in self.letter.items()}
        self.nl = tok.encode("\n", add_special_tokens=False)
        self.end = tok.encode("<|im_end|>", add_special_tokens=False) + self.nl

    def __call__(self, cams, qs, answers=None):
        """answers: list of letters (training) or None (inference: prompt only).
        Returns the processor batch (+ 'labels' when training) and the letter positions."""
        enc = self.p.apply_chat_template(messages(cams, qs), tokenize=True, add_generation_prompt=True,
                                         return_dict=True, return_tensors="pt", enable_thinking=False)
        if answers is None:
            return enc, [enc["input_ids"].shape[1] - 1]
        import torch
        L0 = enc["input_ids"].shape[1]; ids = enc["input_ids"][0].tolist(); pos = []
        for j, L in enumerate(answers):
            pos.append(len(ids) - 1); ids.append(self.letter[L])        # logits at len-1 predict the letter
            if j < len(answers) - 1: ids += self.nl
        ids += self.end
        labels = [-100] * len(ids)
        for p_, L in zip(pos, answers): labels[p_ + 1] = self.letter[L]
        extra = len(ids) - L0
        for k in list(enc.keys()):                    # other per-token arrays (e.g. mm_token_type_ids): text = 0
            v = enc[k]
            if k not in ("input_ids", "attention_mask") and torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == L0:
                enc[k] = torch.cat([v, torch.zeros(1, extra, dtype=v.dtype)], 1)
        enc["input_ids"] = torch.tensor([ids]); enc["attention_mask"] = torch.ones_like(enc["input_ids"])
        enc["labels"] = torch.tensor([labels])
        return enc, pos


class TrainSet:
    """Each draw: a random clip, k in {1,2,3} (weights .2/.3/.5), questions sampled with weights that
    move the training type mix toward the test mix (test share / train share, capped at 10)."""

    def __init__(self, split="train", seed=0):
        """split may join several with '+', e.g. 'train+validation'; clips are (split, clip) pairs."""
        self.qs = {}
        for sp in split.split("+"):
            self.qs.update({(sp, c): v for c, v in load_questions(sp).items()})
        self.clips = [k for k in sorted(self.qs) if clip_images(k[0], k[1]) is not None]
        self.split = split; self.rng = random.Random(seed)
        from collections import Counter
        cnt = Counter(q["subcategory"] for c in self.clips for q in self.qs[c]); n = sum(cnt.values())
        self.w = {s: min(10.0, TEST_SHARE.get(s, 0.0) / (cnt[s] / n)) for s in cnt}
        print(f"[TrainSet] {split}: {len(self.clips)} clips with images, {n} questions; type weights "
              + ", ".join(f"{s[:12]}={w:.2f}" for s, w in sorted(self.w.items())), flush=True)

    def draw(self):
        """-> ((split, clip), questions)"""
        c = self.rng.choice(self.clips); qs = [q for q in self.qs[c] if self.w.get(q["subcategory"], 0) > 0]
        k = min(len(qs), self.rng.choices([1, 2, 3], [.2, .3, .5])[0])
        picked = []
        meta = [q for q in self.qs[c] if q["subcategory"] == "ego_meta_action" and "answer" in q]
        if meta and self.rng.random() < float(os.environ.get("META_P", "0")):
            picked.append(meta[0]); k = max(0, k - 1)
        pool = list(qs)
        for _ in range(k):
            q = self.rng.choices(pool, [self.w[x["subcategory"]] for x in pool])[0]
            picked.append(q); pool.remove(q)
        return c, picked
