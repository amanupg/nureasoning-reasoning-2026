"""Extract features from question TEXT, then learn per-subcategory answers.

Many questions state the object's ego-frame 3D position, its camera, and its
normalized 2D box directly in the prompt. That is enough to answer the spatial
subcategories geometrically, with no image pixels at all. Rather than hand-derive
each mapping we FIT it on the labelled validation VQA, which also measures how
well it works.
"""
import json, math, os, re, collections
import numpy as np

RE_BOX = re.compile(r'normalized 2D box \[([-\d\s,\.]+)\]')
RE_POS = re.compile(r'ego-frame 3D position \[x, y\] = \[\s*(-?[\d\.]+),\s*(-?[\d\.]+)\s*\]')
RE_CAM = re.compile(r'in the ([a-z ]+?) camera')
RE_RANGE = re.compile(r'at (near|mid|far) range')
RE_DIR = re.compile(r'to the ([a-z\-]+) of the ego|directly (in front of|behind)')
CAMS = ["front", "front_left", "front_right", "left", "right",
        "back", "back_left", "back_right"]
# "What is the risk level of: 'Maintain speed' (longitudinal) + 'No lateral action' (lateral)?"
RE_ACT = re.compile(r"'([^']+)'\s*\(longitudinal\)\s*\+\s*'([^']+)'\s*\(lateral\)")


def norm_text(s):
    return re.sub(r"\s+", " ", str(s).strip().lower().replace("_", " "))


def feats(q):
    t = q.get("question", "")
    f = {}
    mp = RE_POS.search(t)
    if mp:
        x, y = float(mp.group(1)), float(mp.group(2))
        f["x"], f["y"] = x, y
        f["r"] = math.hypot(x, y)
        f["az"] = math.degrees(math.atan2(y, x))     # +left, 0 = straight ahead
    mb = RE_BOX.search(t)
    if mb:
        v = [float(z) for z in mb.group(1).split(",")]
        if len(v) == 4:
            x0, y0, x1, y1 = v
            f["bx"] = (x0 + x1) / 2.0
            f["by"] = (y0 + y1) / 2.0
            f["bw"] = abs(x1 - x0)
            f["bh"] = abs(y1 - y0)
            f["barea"] = f["bw"] * f["bh"]
    mc = RE_CAM.search(t)
    if mc:
        f["cam"] = mc.group(1).strip().replace(" ", "_")
    mr = RE_RANGE.search(t)
    if mr:
        f["range"] = mr.group(1)
    md = RE_DIR.search(t)
    if md:
        f["dir"] = (md.group(1) or md.group(2) or "").strip()
    ma = RE_ACT.search(t)
    if ma:
        f["act_lon"] = ma.group(1).strip().lower()
        f["act_lat"] = ma.group(2).strip().lower()
    # object noun, e.g. "For the car in the front camera"
    mo = re.search(r'\b(?:the|this)\s+([a-z_ ]+?)\s+(?:object\s+)?(?:in|that|with)\b', t)
    if mo:
        f["obj"] = mo.group(1).strip()
    return f


def load(root="vqa_val"):
    out = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        for fn in sorted(os.listdir(p)):
            if not fn.endswith(".json"):
                continue
            o = json.load(open(os.path.join(p, fn)))
            qq = o.get("questions") if isinstance(o, dict) else o
            for q in (qq or []):
                if q.get("question_type") == "choice" and q.get("answer"):
                    q["_clip"] = d
                    q["_f"] = feats(q)
                    out.append(q)
    return out


if __name__ == "__main__":
    qs = load()
    print(f"labelled: {len(qs)}")
    cov = collections.defaultdict(collections.Counter)
    for q in qs:
        s = q.get("subcategory"); f = q["_f"]
        for k in ("x", "bx", "cam", "range", "dir", "obj"):
            cov[s][k] += (k in f)
        cov[s]["n"] += 1
    print(f"\n{'subcategory':34} {'n':>6} {'xy':>6} {'bbox':>6} {'cam':>6} {'rng':>6} {'dir':>6} {'obj':>6}")
    for s in sorted(cov):
        c = cov[s]; n = c["n"]
        print(f"{s:34} {n:>6} " + " ".join(f"{100*c[k]/n:>5.0f}%" for k in
                                           ("x", "bx", "cam", "range", "dir", "obj")))
