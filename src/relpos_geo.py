"""Geometric answer for object_relative_position / ego_position_from_2d_bbox (inputs only: camera, 2D box,
category, calibration). Public-generator definition (validation data): front if x > 1.5, rear if x < -1.5, else
side_by_side; left if y > 1.5, right if y < -1.5, else aligned (aligned -> 'directly in front of' / 'directly behind').
Position = camera ray through the box centre, depth from box height and a per-category object height.
usage: python src/relpos_geo.py <split>  -> artifacts/relpos_geo_<split>.json {qid: {"pred": letter or None, "x":, "y":}}"""
import glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camgeo
R = re.compile(r"the ([\w ]+?) (?:object )?in the (.+?) camera with normalized 2D box \[(\d+), (\d+), (\d+), (\d+)\]")
H = {"car": 1.65, "truck": 3.0, "bus": 3.2, "human": 1.7, "pedestrian": 1.7, "traffic cone": 0.75, "traffic_cone": 0.75,
     "bicycle": 1.4, "motorcycle": 1.4, "barrier": 1.0, "vehicle": 1.8}
SUBS = ("object_relative_position", "ego_position_from_2d_bbox")


def label(x, y):
    fr = "front" if x > 1.5 else ("rear" if x < -1.5 else "side")
    lr = "left" if y > 1.5 else ("right" if y < -1.5 else "aligned")
    if fr == "side": return {"left": "to the left of", "right": "to the right of"}.get(lr)
    if lr == "aligned": return "in front of" if fr == "front" else "behind"
    return f"to the {fr}-{lr} of"


def solve(rig, q):
    m = R.search(q["question"])
    if not m: return None
    cat, cam = m.group(1), m.group(2).replace(" ", "_")
    if cam not in rig.cams: return None
    x1, y1, x2, y2 = [int(v) for v in m.groups()[2:6]]
    c = rig.cams[cam]; hpx = max(1.0, (y2 - y1) / 1000 * c["H"])
    if y1 <= 2 or y2 >= 998: hpx = None                  # truncated vertically: height unreliable
    d, t = rig.ray(cam, (x1 + x2) / 2, (y1 + y2) / 2)
    ax = c["R"] @ np.array([0, 0, 1.0])
    depth = (c["K"][1, 1] * H.get(cat, 1.65) / hpx) if hpx else 15.0
    p = t + d * depth / max(1e-3, float(d @ ax))
    return float(p[0]), float(p[1])


def norm(s): return s.replace("directly ", "")


def main():
    split = sys.argv[1]
    if split == "test":
        Q = {f.split("/")[2]: json.load(open(f))["questions"] for f in glob.glob("cache/test/*/reasoning_questions.json")}
    else:
        Q = {}
        for f in glob.glob("vqa_val/*/*_vqa.json"):
            Q.setdefault(f.split("/")[1], []).extend(json.load(open(f)).get("questions") or [])
    out = {}; ok = n = 0
    for clip, qs in Q.items():
        qs = [q for q in qs if q.get("subcategory") in SUBS and q.get("choices")]
        if not qs: continue
        rig = camgeo.Rig(json.load(open(f"cache/{split}/{clip}/metadata.json")))
        for q in qs:
            r = solve(rig, q)
            if r is None: continue
            lab = label(*r); pred = None
            if lab:
                hits = [L for L, v in q["choices"].items() if norm(v) == lab]
                pred = hits[0] if hits else None
            out[q["question_id"]] = {"pred": pred, "x": r[0], "y": r[1], "sub": q["subcategory"], "label": lab}
            if q.get("answer") is not None:
                n += 1; ok += pred == q["answer"]
    json.dump(out, open(f"artifacts/relpos_geo_{split}.json", "w"))
    print(f"{split}: {len(out)} questions; with gold {n}, geometry accuracy {ok / max(n, 1):.3f}; "
          f"no matching option {sum(o['pred'] is None for o in out.values())}")


if __name__ == "__main__":
    main()
