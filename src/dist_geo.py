"""distance_from_2d_bbox from calibration geometry (generator-independent).

Two physical estimates from the stated camera + normalised 2D box:
  ground : ray through the box's bottom-centre pixel meets the ground plane z = zg
  size   : class height H over pixel height, d = fy * H / h_px (along the ray)
Fitted on half the validation DRIVE LOGS (zg, per-class H, blend), scored on the
other half. Answer = choice nearest the estimate in log distance.
"""
import collections, json, math, os, random, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import camgeo
from reason_model import load, num_of, norm_text
from qfeat import RE_BOX

SUB = "distance_from_2d_bbox"
_RIG = {}


def rig(clip, root):
    if clip not in _RIG:
        _RIG[clip] = camgeo.Rig(json.load(open(os.path.join(root, clip, "metadata.json"))))
    return _RIG[clip]


def raw(q, root):
    """Geometry primitives for one question (independent of fitted parameters)."""
    f = q["_f"]; cam = f.get("cam"); mb = RE_BOX.search(q["question"])
    if not mb or cam is None: return None
    R = rig(q["_clip"], root)
    if cam not in R.cams: return None
    x0, y0, x1, y1 = [float(v) for v in mb.group(1).split(",")]
    c = R.cams[cam]
    d, o = R.ray(cam, (x0 + x1) / 2, max(y0, y1))
    dc, _ = R.ray(cam, (x0 + x1) / 2, (y0 + y1) / 2)
    hpx = abs(y1 - y0) / 1000 * c["H"]
    return dict(d=d, o=o, dc=dc, ax=c["R"] @ np.array([0.0, 0.0, 1.0]), hpx=max(hpx, 1.0), fy=c["K"][1, 1], obj=f.get("obj", "?"),
                trunc=max(y0, y1) >= 995 or min(y0, y1) <= 5)


def est_ground(p, zg):
    d, o = p["d"], p["o"]
    if d[2] >= -1e-3: return None
    t = (zg - o[2]) / d[2]
    P = o + t * d
    return math.hypot(P[0], P[1])


def est_size(p, H):
    # depth along the optical axis -> distance along the centre ray, then to ego origin
    depth = p["fy"] * H / p["hpx"]                       # along the optical axis
    along = depth / max(float(np.dot(p["dc"], p["ax"])), 0.2)   # along the box-centre ray
    P = p["o"] + along * p["dc"]
    return math.hypot(P[0], P[1])


def pick(q, est):
    ch = {k: num_of(v) for k, v in (q.get("choices") or {}).items()}
    ch = {k: v for k, v in ch.items() if v is not None and v > 0}
    if not ch or est is None or est <= 0: return None
    return min(ch, key=lambda k: abs(math.log(ch[k]) - math.log(est)))


def main(root="cache/validation"):
    qs = [q for q in load() if q.get("subcategory") == SUB]
    for q in qs: q["_p"] = raw(q, root); q["_y"] = num_of(q.get("answer_text"))
    qs = [q for q in qs if q["_p"] is not None and q["_y"]]
    log_of = lambda c: "_".join(c.split("_")[:2])
    logs = sorted({log_of(q["_clip"]) for q in qs}); random.Random(0).shuffle(logs)
    half = set(logs[:len(logs) // 2])
    fit = [q for q in qs if log_of(q["_clip"]) in half]; ev = [q for q in qs if log_of(q["_clip"]) not in half]
    acc = lambda rows, fn: np.mean([pick(q, fn(q)) == q["answer"] for q in rows])

    # ground height: maximise fit accuracy of the ground estimate
    zs = np.arange(-2.6, 0.61, 0.05)
    za = [(acc(fit, lambda q, z=z: est_ground(q["_p"], z)), z) for z in zs]
    zg = max(za)[1]
    # per-class height: median of the implied height on fit rows (untruncated boxes)
    Hs = collections.defaultdict(list)
    for q in fit:
        p = q["_p"]
        if p["trunc"]: continue
        # distance scales linearly with H in est_size, so implied H = y / est_size(H=1)
        s1 = est_size(p, 1.0)
        if s1 > 0: Hs[p["obj"]].append(q["_y"] / s1)
    H = {k: float(np.median(v)) for k, v in Hs.items() if len(v) >= 10}
    Hdef = float(np.median([h for v in Hs.values() for h in v]))
    size = lambda q: est_size(q["_p"], H.get(q["_p"]["obj"], Hdef))

    def blend(q, w):
        g, s = est_ground(q["_p"], zg), size(q)
        if q["_p"]["trunc"] and g: return g
        if g is None or g <= 0 or g > 150: return s
        return math.exp(w * math.log(g) + (1 - w) * math.log(s))
    wb = max(((acc(fit, lambda q, w=w: blend(q, w)), w) for w in np.arange(0, 1.01, .1)))[1]

    print(f"{SUB}: fit {len(fit)}  eval {len(ev)}  (split by drive log)")
    print(f"  ground plane zg = {zg:.2f} m   class heights { {k: round(v, 2) for k, v in sorted(H.items())} }")
    print(f"  eval accuracy  ground {acc(ev, lambda q: est_ground(q['_p'], zg)):.3f}   size {acc(ev, size):.3f}   "
          f"blend(w={wb:.1f}) {acc(ev, lambda q: blend(q, wb)):.3f}   [qmodel 0.694]")
    by = collections.defaultdict(list)
    for q in ev: by[q["_p"]["obj"]].append(pick(q, blend(q, wb)) == q["answer"])
    print("  by class:", {k: (len(v), round(float(np.mean(v)), 3)) for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))[:8]})
    out = {"zg": zg, "H": H, "H_default": Hdef, "w_ground": wb, "fitted_on": "vqa_val half by drive log",
           "eval_acc": float(acc(ev, lambda q: blend(q, wb)))}
    json.dump(out, open("artifacts/dist_geo.json", "w"), indent=1)
    print("  -> artifacts/dist_geo.json")


if __name__ == "__main__":
    main()


def geo_feats(q, root):
    """Features for the distance regressor: log ground-plane and log unit-height size
    estimates from this clip's own calibration. Empty for other subcategories, so
    their models see all-NaN columns, which usable_cols drops (models unchanged)."""
    if q.get("subcategory") != SUB:
        return {}
    try:
        p = raw(q, root)
    except Exception:
        return {}
    if p is None:
        return {}
    g = est_ground(p, 0.0); s = est_size(p, 1.0)
    return {"g_ground": math.log(g) if g and g > 0 else float("nan"),
            "g_size": math.log(s) if s > 0 else float("nan")}
