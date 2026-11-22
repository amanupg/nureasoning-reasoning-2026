"""2026-11-14 dense-frame training data for the VLM (spec ../reports/dense_vlm_run_spec.md). A copy-and-extend of
src/vlm_data.py's TrainSet (the original stays untouched; Encoder / messages / prompt format are imported from it).
- One example = one (clip, frame): key frame 100 (questions vqa_train_cl, images img_vlm) or an extra annotated frame
  (questions vqa_train_dense_cl from src/gen_vqa_dense.py, images img_dense_A / img_dense_B from the fetch passes).
  Images: the 8 cameras at frame t (key) and t-10 (1 s earlier), resolved through the clip's metadata.json.
- Only (clip, frame) pairs whose 16 images are on disk are used (checked once at start, so a partial fetch works).
- 11-15: questions from one packed file (src/dense_qpack.py) and image paths from a prebuilt index
  (src/dense_index.py); the first version read ~13 GB of metadata.json on GPFS at start and never finished.
- k questions per example in {1, 3, 6} with weights .15/.35/.50; never two questions with the same target_track_token.
- Type targets (share of question draws): distance .20, future_motion .17, image_region .13, action_risk .12,
  camera_view .08, each driving decision .04, unsafe .04, the other seven test types .14 split by test share.
  Per-question weight = target share / pooled train share of the type.
FOCUS / EGO are not used here (EGO features exist only at the key frame).
Use: DENSE=1 in src/vlm_sft.py (imports TrainSet and clip_images from this module)."""
import json, math, os, random
from collections import Counter, defaultdict
from vlm_data import TEST_SHARE

TARGET = {"distance_from_2d_bbox": .20, "future_motion_label": .17, "image_region_from_3d_object": .13,
          "action_risk_assessment": .12, "camera_view_for_3d_object": .08, "driving_decision_joint": .04,
          "driving_decision_lateral": .04, "driving_decision_longitudinal": .04, "unsafe_action_identification": .04}
_rest = {s: v for s, v in TEST_SHARE.items() if s not in TARGET}
TARGET.update({s: .14 * v / sum(_rest.values()) for s, v in _rest.items()})
KS, KW = (1, 3, 6), (.15, .35, .50)
if os.environ.get("DENSE_KS"):  # 2026-11-15: questions per example, e.g. DENSE_KS=1,6,12 DENSE_KW=.15,.35,.5
    KS = tuple(int(x) for x in os.environ["DENSE_KS"].split(",")); KW = tuple(float(x) for x in os.environ["DENSE_KW"].split(","))
ROOTS = [r for r in os.environ.get("DENSE_IMG_ROOTS", "img_vlm:img_dense_A:img_dense_B").split(":") if r]
_IDX = None                       # artifacts/dense_index_train.json (src/dense_index.py): no metadata reads on GPFS


def _index():
    global _IDX
    if _IDX is None: _IDX = json.load(open(os.environ.get("DENSE_INDEX", "artifacts/dense_index_train.json")))   # 11-16: v2 adds pass C
    return _IDX


def frame_images(split, clip, t):
    """[(cam name, t-10 path, t path)] from the prebuilt index (root chosen by fetch pass; no per-image stat)."""
    e = _index().get(clip, {}).get("img", {}).get(str(t))
    if e is None: return None
    root, cams = e
    return [(name, f"{root}/{split}/{clip}/{a}", f"{root}/{split}/{clip}/{b}") for name, a, b in cams]


def clip_images(split, key):
    clip, t = key
    return frame_images(split, clip, t)


def _available():
    """(clip, frame) pairs whose 16 images are on disk, from the fetch passes' per-clip _done.json (written only when
    every member of that clip's pass arrived) and the img_vlm key-frame set; cheap on GPFS (no per-image stat)."""
    out = set()
    if os.path.isdir("img_vlm/train"):
        out |= {(c, 100) for c in os.listdir("img_vlm/train")}
    for tag in tuple(os.environ.get("DENSE_PASSES", "AB")):      # 11-16: DENSE_PASSES=ABC adds pass C (all 13 annotated frames)
        mp, root = f"artifacts/dense_members_train_{tag}.json", f"img_dense_{tag}/train"
        if not (os.path.isfile(mp) and os.path.isdir(root)): continue
        done = {c for c in os.listdir(root) if os.path.isfile(f"{root}/{c}/_done.json")}
        out |= {(r["clip"], t) for r in json.load(open(mp))["train"] if r["clip"] in done for t in r["frames"]}
    return out


def _choice(qs): return [q for q in qs if q.get("question_type", "choice") == "choice" and q.get("choices") and "answer" in q]


class TrainSet:
    def __init__(self, split="train", seed=0):
        assert split == "train", "dense data exists for train only"
        self.rng = random.Random(seed); ex = defaultdict(list); n_dense = 0
        for l in open(os.environ.get("DENSE_QPACK", "artifacts/dense_q_train.jsonl")):   # src/dense_qpack.py
            r = json.loads(l); ex[(r["clip"], r["frame"])] += _choice(r["qs"]); n_dense += r["frame"] != 100
        idx = _index()
        ex = {k: v for k, v in ex.items() if str(k[1]) in idx.get(k[0], {}).get("img", {})}
        avail = _available()
        self.keys = sorted(k for k, v in ex.items() if v and k in avail)
        self.qs = {k: ex[k] for k in self.keys}
        cnt = Counter(q["subcategory"] for k in self.keys for q in self.qs[k]); n = sum(cnt.values())
        self.w = {s: TARGET.get(s, 0.0) / (cnt[s] / n) for s in cnt}
        nf = Counter(k[1] == 100 for k in self.keys)
        print(f"[TrainSet dense] {len(self.keys)} (clip, frame) pairs with all 16 images ({nf[True]} key, {nf[False]} extra; "
              f"{len({k[0] for k in self.keys})} clips; dense frames with questions {n_dense}); {n} questions; type weights "
              + ", ".join(f"{s[:12]}={w:.2f}" for s, w in sorted(self.w.items())), flush=True)
        self.split = "train"
        self._calibrate()

    def _calibrate(self, iters=int(os.environ.get("DENSE_CAL_ITERS", 12)), n=4000):
        """Type-first sampling (11-15): uniform frame sampling starved scarce types (smoke 13912887: the three decision
        types together 3.7% of draws vs 12% target, since most frames have no decision questions). Draw the first
        question's type from p, a frame holding that type, then fill k-1 from the same frame by self.w. p is tuned by
        simulation so the realised share of every question drawn matches TARGET (ratio clipped to [0.2, 5] per step)."""
        self.by_type = defaultdict(list)
        for key in self.keys:
            for s in {q["subcategory"] for q in self.qs[key]}: self.by_type[s].append(key)
        types = [s for s in TARGET if self.by_type.get(s)]
        tot = sum(TARGET[s] for s in types); tgt = {s: TARGET[s] / tot for s in types}
        # 2026-11-15: fill acceptance self.acc[s] <= 1. At k=12 the fill ran out of hard questions and padded examples
        # with easy types (object_relative_position 0.107 realised vs 0.023 target). A fill question of an over-drawn type
        # is now skipped with probability 1 - acc[s] (examples get shorter instead of padded), calibrated with p.
        self.p = dict(tgt); self.acc = {s: 1.0 for s in types}; rng0 = self.rng; self.rng = random.Random(12345)
        for it in range(iters):
            T = Counter(); nk = 0
            for _ in range(n): qs = self.draw()[1]; T.update(q["subcategory"] for q in qs); nk += len(qs)
            m = sum(T.values()); r = {s: max(T[s] / m, 1e-6) for s in types}
            self.p = {s: self.p[s] * min(5, max(.2, tgt[s] / r[s])) for s in types}
            z = sum(self.p.values()); self.p = {s: v / z for s, v in self.p.items()}
            self.acc = {s: min(1.0, self.acc[s] * min(2, max(.2, tgt[s] / r[s]))) for s in types}
            if it == iters - 1 or os.environ.get("DENSE_CAL_VERBOSE"):
                worst = max(types, key=lambda s: abs(math.log(r[s] / tgt[s])))
                print(f"[calibrate {it}] mean k {nk / n:.2f}, worst {worst} {r[worst]:.3f} vs {tgt[worst]:.3f}", flush=True)
        self.rng = rng0
        print("[TrainSet dense] fill acceptance: " + ", ".join(f"{s[:22]}={v:.2f}" for s, v in sorted(self.acc.items())), flush=True)
        print("[TrainSet dense] type-first p: " + ", ".join(f"{s[:22]}={v:.3f}" for s, v in sorted(self.p.items())), flush=True)

    def draw(self):
        """-> (("train", (clip, frame)), questions). First question's type ~ self.p, then fill from the same frame."""
        s = self.rng.choices(list(self.p), list(self.p.values()))[0]
        key = self.rng.choice(self.by_type[s])
        pool = [q for q in self.qs[key] if self.w.get(q["subcategory"], 0) > 0]
        first = self.rng.choice([q for q in pool if q["subcategory"] == s])
        pool.remove(first); picked = [first]; used = {first["target_track_token"]} if first.get("target_track_token") else set()
        k = self.rng.choices(KS, KW)[0]
        while pool and len(picked) < k:
            q = self.rng.choices(pool, [self.w[x["subcategory"]] for x in pool])[0]
            pool.remove(q); tok = q.get("target_track_token")
            if tok and tok in used: continue
            if self.rng.random() >= getattr(self, "acc", {}).get(q["subcategory"], 1.0): continue
            picked.append(q)
            if tok: used.add(tok)
        return ("train", key), picked


if __name__ == "__main__":                                  # smoke: realised k / type mix / frames over 2,000 draws
    d = TrainSet("train", seed=0); K = Counter(); T = Counter(); F = Counter(); clash = 0; seen = set()
    for _ in range(2000):
        (_, key), qs = d.draw(); seen.add(key); K[len(qs)] += 1; F[key[1] == 100] += 1; T.update(q["subcategory"] for q in qs)
        toks = [q.get("target_track_token") for q in qs if q.get("target_track_token")]; clash += len(toks) != len(set(toks))
    n = sum(T.values())
    print("k", dict(sorted(K.items())), "key-frame share", round(F[True] / 2000, 3), "token clashes", clash)
    print("type mix vs target", {s: (round(v / n, 3), round(TARGET.get(s, 0), 3)) for s, v in T.most_common()})
    print("distinct (clip, frame) drawn", len(seen), "of", len(d.keys))
