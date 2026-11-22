"""future_motion_label from tracked boxes (fm_track.py output): range from box scale, closing speed,
the fixed 0.5 m/s rule. No fitting to labels.

Geometry per question (clip's own calibration, camgeo conventions, lidar frame = ego frame):
  frame 100: ray through the stated box centre; the point on it whose ego (x, y) is closest to the
  stated position -> camera depth z0 (exact range anchor).
  frame k:   z_k = z0 * s(100) / s(k), s = a size measure of the tracked mask box (height, width,
  sqrt(area), ...); position = the ray through the tracked box centre at depth z_k (ego frame at k).
  range r_k = |(x, y)|; closing speed = -dr/dt (Theil-Sen slope over the usable frames, camera
  timestamps). Range relative to the ego origin needs no ego poses: |p| is rotation invariant.
Rule: closing > 0.5 approaching, < -0.5 moving away, else maintaining.

usage: python src/fm_rule.py <split> <track_prefix> [--truth]   (--truth: validation, annotated closing speed)
"""
import argparse, glob, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import camgeo

LAB = ("approaching ego", "moving away from ego", "maintaining relative distance")


def rule(c, t=0.5):
    return LAB[0] if c > t else (LAB[1] if c < -t else LAB[2])


def theil_sen(t, r):
    s = [(r[j] - r[i]) / (t[j] - t[i]) for i in range(len(t)) for j in range(i + 1, len(t)) if abs(t[j] - t[i]) > 1e-3]
    return float(np.median(s)) if s else float("nan")


def size(b, area, how):
    h, w = b[3] - b[1], b[2] - b[0]
    return {"h": h, "w": w, "hw": math.sqrt(max(h, 1e-6) * max(w, 1e-6)), "area": math.sqrt(max(area, 1e-12))}[how]


def point_at_depth(c, u, v, z):
    pc = np.linalg.inv(c["K"]) @ np.array([u / 1000 * c["W"], v / 1000 * c["H"], 1.0])
    return c["R"] @ (pc * z / pc[2]) + c["t"]


def closing(rec, rig, how="h", edge=3.0, min_frames=3):
    c = rig.cams.get(rec["cam"])
    if c is None or rec.get("missing"): return float("nan"), "nocam"
    bq = rec["box_q"]; uc, vc = (bq[0] + bq[2]) / 2, (bq[1] + bq[3]) / 2
    d, o = rig.ray(rec["cam"], uc, vc); dxy = d[:2]; p = np.array([rec["x"], rec["y"]])
    s0 = float(dxy @ (p - o[:2]) / max(dxy @ dxy, 1e-9))
    z0 = float((c["R"].T @ (o + s0 * d - c["t"]))[2])
    if z0 <= 0.5: return float("nan"), "behind"
    B, A = rec["box"], rec["area"]
    if B[0] is None: return float("nan"), "nomask100"
    ref = size(B[0], A[0], how)
    ts, rs = [], []
    for i, b in enumerate(B):
        if b is None or (rec["score"][i] is not None and rec["score"][i] < 0): continue
        if min(b[0], b[1]) < edge or max(b[2], b[3]) > 1000 - edge: continue          # truncated at the image border
        z = z0 * ref / max(size(b, A[i], how), 1e-6)
        P = point_at_depth(c, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, z)
        ts.append(rec["t"][i]); rs.append(math.hypot(P[0], P[1]))
    if len(ts) < min_frames or (max(ts) - min(ts)) < 0.25: return float("nan"), "fewframes"
    return -theil_sen(ts, rs), "ok"


def truth(split, recs):
    """Annotated closing speed of each question's object (nearest relation at frame ~100, <= 0.3 m)."""
    def walk(o):
        if isinstance(o, dict):
            if "future_movement_description" in o: yield o
            for v in o.values(): yield from walk(v)
        elif isinstance(o, list):
            for v in o: yield from walk(v)
    rel_cache, out = {}, {}
    for qid, r in recs.items():
        c = r["clip"]
        if c not in rel_cache:
            rel_cache[c] = [rel for fn in glob.glob(f"cache/{split}/{c}/reasoning/*.json")
                            for R in [json.load(open(fn))] if abs(R["frame_index"] - 100) <= 1 for rel in walk(R)]
        best = min(rel_cache[c], key=lambda rel: math.hypot(rel["position_3d_ego"]["x"] - r["x"], rel["position_3d_ego"]["y"] - r["y"]), default=None)
        if best and math.hypot(best["position_3d_ego"]["x"] - r["x"], best["position_3d_ego"]["y"] - r["y"]) <= 0.3:
            out[qid] = best["geometric_relations"]["closing_speed_mps"]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("split"); ap.add_argument("prefix"); ap.add_argument("--truth", action="store_true")
    a = ap.parse_args()
    recs = {}
    for f in sorted(glob.glob(f"{a.prefix}_s*of*.json")): recs.update(json.load(open(f)))
    rigs = {}
    T = truth(a.split, recs) if a.truth else {}
    print(f"{len(recs)} tracked questions; truth matched {len(T)}")
    for how in ("h", "w", "hw", "area"):
        est, why = {}, {}
        for qid, r in recs.items():
            if r["clip"] not in rigs: rigs[r["clip"]] = camgeo.Rig(json.load(open(f"cache/{a.split}/{r['clip']}/metadata.json")))
            est[qid], why[qid] = closing(r, rigs[r["clip"]], how)
        ok = [q for q in recs if not math.isnan(est[q])]
        from collections import Counter
        line = f"[{how:4s}] usable {len(ok)}/{len(recs)} {dict(Counter(why.values()))}"
        if a.truth:
            q2 = [q for q in ok if q in T]; e = np.array([est[q] for q in q2]); g = np.array([T[q] for q in q2])
            acc = np.mean([rule(est[q]) == recs[q]["answer"] for q in q2])
            line += f" | corr {np.corrcoef(e, g)[0,1]:.3f} med|err| {np.median(np.abs(e-g)):.2f} m/s rule acc {acc:.3f} (n={len(q2)})"
            for cam in sorted({recs[q]["cam"] for q in q2}):
                qq = [q for q in q2 if recs[q]["cam"] == cam]
                line += f"\n        {cam:12s} n={len(qq):4d} acc {np.mean([rule(est[q]) == recs[q]['answer'] for q in qq]):.3f} med|err| {np.median([abs(est[q]-T[q]) for q in qq]):.2f}"
        print(line)
        json.dump({q: est[q] for q in ok}, open(f"{a.prefix}_closing_{how}.json", "w"))


if __name__ == "__main__":
    main()
