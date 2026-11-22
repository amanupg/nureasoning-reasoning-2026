"""Samples for the box-conditioned 3D regressor (distance and closing speed from a 2D detector box).

Annotation samples (train / validation): every object in the frame-100 reasoning annotation, once per
camera that observed it (detection_bbox_2d = the same Gemini-detector boxes the questions quote),
normalised to 0..1000 with that camera's calibrated size. Targets: ego-frame x, y, dist = hypot(x, y),
closing_speed_mps (the exact source of the future-motion label), velocity.
Question samples (validation / test): distance_from_2d_bbox and future_motion_label questions parsed
from their text (camera, box, category, stated x/y when given).
-> artifacts/box3d_{ann,q}_<split>.jsonl
usage: python src/box3d_data.py <split> [ann|q|both]
"""
import glob, json, os, re, sys

RE = re.compile(r"For the (.+?) in the ([a-z ]+?) camera with normalized 2D box \[([-\d\s,\.]+)\]")
RP = re.compile(r"\[x, y\]\s*=\s*\[\s*(-?[\d.]+),\s*(-?[\d.]+)\s*\]")
KEY = {"front": "CAM_M_F", "front_left": "CAM_M_L0", "front_right": "CAM_M_R0", "left": "CAM_M_L1",
       "right": "CAM_M_R1", "back_left": "CAM_M_L2", "back_right": "CAM_M_R2", "back": "CAM_M_B"}
SUBS = ("distance_from_2d_bbox", "future_motion_label")


def walk(o):
    if isinstance(o, dict):
        if "camera_observations" in o and "position_3d_ego" in o: yield o
        for v in o.values(): yield from walk(v)
    elif isinstance(o, list):
        for v in o: yield from walk(v)


def ann_samples(split):
    out = []
    for cp in sorted(glob.glob(f"cache/{split}/*")):
        mp = cp + "/metadata.json"
        if not os.path.isfile(mp): continue
        meta = json.load(open(mp)); fr = meta["frames"]
        rel = fr[100].get("reasoning") if len(fr) > 100 else None
        if not rel or not os.path.isfile(os.path.join(cp, rel)): continue
        cal = meta.get("camera_calibrations") or {}; clip = os.path.basename(cp)
        for o in walk(json.load(open(os.path.join(cp, rel)))):
            p, v = o["position_3d_ego"], o.get("velocity_3d_ego") or {}
            cs = (o.get("geometric_relations") or {}).get("closing_speed_mps")
            for cam, ob in o["camera_observations"].items():
                b = ob.get("detection_bbox_2d"); c = cal.get(KEY.get(cam, ""), {})
                if not b or not c: continue
                W, H = float(c["width"]), float(c["height"])
                box = [1000 * b[0] / W, 1000 * b[1] / H, 1000 * b[2] / W, 1000 * b[3] / H]
                out.append({"split": split, "clip": clip, "cam": cam, "box": [round(x, 1) for x in box],
                            "cat": o["category"].split(".")[-1], "x": p["x"], "y": p["y"],
                            "dist": (p["x"] ** 2 + p["y"] ** 2) ** 0.5, "closing": cs,
                            "vx": v.get("x"), "vy": v.get("y"), "track": o.get("track_token")})
    return out


def q_samples(split):
    if split == "test":
        files = [(f.split("/")[2], f) for f in glob.glob("cache/test/*/reasoning_questions.json")]
    else:
        files = [(f.split("/")[1], f) for f in glob.glob("vqa_val/*/*_vqa.json")]
    out = []
    for clip, f in files:
        for q in json.load(open(f)).get("questions") or []:
            if q.get("subcategory") not in SUBS: continue
            m = RE.search(q["question"])
            if not m: continue
            mp = RP.search(q["question"])
            out.append({"split": split, "clip": clip, "cam": m.group(2).replace(" ", "_"),
                        "box": [float(x) for x in m.group(3).split(",")], "cat": m.group(1).split(".")[-1],
                        "qid": q["question_id"], "sub": q["subcategory"], "choices": q["choices"],
                        "answer": q.get("answer"), "x": float(mp.group(1)) if mp else None,
                        "y": float(mp.group(2)) if mp else None})
    return out


if __name__ == "__main__":
    split = sys.argv[1]; what = sys.argv[2] if len(sys.argv) > 2 else "both"
    if what in ("ann", "both") and split != "test":
        A = ann_samples(split)
        with open(f"artifacts/box3d_ann_{split}.jsonl", "w") as f:
            for r in A: f.write(json.dumps(r) + "\n")
        print(split, "annotation samples", len(A))
    if what in ("q", "both") and split != "train":
        Q = q_samples(split)
        with open(f"artifacts/box3d_q_{split}.jsonl", "w") as f:
            for r in Q: f.write(json.dumps(r) + "\n")
        print(split, "question samples", len(Q))
