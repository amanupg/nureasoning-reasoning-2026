"""(Monocular distance helpers, 2026-10-03; formerly misnamed dist_geo.py.) distance_from_2d_bbox: label = hypot(x, y) of the object's centre in the ego frame (euclidean_distance_m,
exact). Choices = the true distance + 3 other observed objects' distances from the same scene.
Measurement-only estimators from the stated box + the clip's calibration; pick the nearest choice (log).
  ground : ray through the box bottom centre hits the ground plane (z = 0 in the lidar/ego frame)
  height : depth from box height and a per-category object height prior (median over TRAIN annotations)
-> artifacts/dist_geo_<split>.json {question_id: {ground, height, cat, cam, box}}
usage: python src/dist_geo.py <split> [--truth]   (validation: accuracy vs labels + noise-tolerance table)
"""
import argparse, glob, json, math, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camgeo
RE = re.compile(r"For the (.+?) in the ([a-z ]+?) camera with normalized 2D box \[([-\d\s,\.]+)\]")


def questions(split):
    if split == "test":
        files = [(f.split("/")[2], f) for f in glob.glob("cache/test/*/reasoning_questions.json")]
    else:
        files = [(f.split("/")[1], f) for f in glob.glob("vqa_val/*/*_vqa.json")]
    out = []
    for clip, f in files:
        J = json.load(open(f)); qs = J if isinstance(J, list) else J.get("questions", [])
        for q in qs:
            if q.get("subcategory") != "distance_from_2d_bbox": continue
            m = RE.search(q["question"])
            if not m: continue
            out.append(dict(q=q, clip=clip, cat=m.group(1), cam=m.group(2).replace(" ", "_"),
                            box=[float(v) for v in m.group(3).split(",")]))
    return out


def ground(rig, cam, u, v):
    d, o = rig.ray(cam, u, v)
    if d[2] >= -1e-4: return float("nan")
    p = o + (-o[2] / d[2]) * d
    return math.hypot(p[0], p[1])


def by_height(rig, cam, box, H):
    c = rig.cams[cam]; fy = c["K"][1, 1]; hpx = (box[3] - box[1]) / 1000 * c["H"]
    if hpx < 2 or not H: return float("nan")
    z = fy * H / hpx                                         # camera depth
    pc = np.linalg.inv(c["K"]) @ np.array([(box[0] + box[2]) / 2000 * c["W"], (box[1] + box[3]) / 2000 * c["H"], 1.0])
    P = c["R"] @ (pc * z / pc[2]) + c["t"]
    return math.hypot(P[0], P[1])


def pick(est, choices):
    if not np.isfinite(est) or est <= 0: return None
    val = {k: float(v.split()[0]) for k, v in choices.items()}
    return min(val, key=lambda k: abs(math.log(max(val[k], 0.1) / est)))
