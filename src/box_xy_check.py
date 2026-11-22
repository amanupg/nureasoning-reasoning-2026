"""Do the stated 2D box and the stated ego-frame [x, y] of the same object agree? (inputs only: question text
+ calibration). Horizontal check: the projected bearing of [x, y] (z from 0.3 to 1.5 m) vs the box's column
range. Reports, per split, the share of questions whose projection falls inside the box (with margin) and
the signed offset (projection - box centre, in box widths).
usage: python src/box_xy_check.py"""
import glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camgeo
R = re.compile(r"in the (.+?) camera with normalized 2D box \[(\d+), (\d+), (\d+), (\d+)\].*?\[x, y\] = \[(-?[\d.]+), (-?[\d.]+)\]")


def qs(split):
    if split == "test":
        for f in glob.glob("cache/test/*/reasoning_questions.json"):
            for q in json.load(open(f)).get("questions") or []: yield f.split("/")[2], q
    else:
        for f in glob.glob("vqa_val/*/*_vqa.json"):
            for q in json.load(open(f)).get("questions") or []: yield f.split("/")[1], q


for split in ("validation", "test"):
    rows = []; rigs = {}
    for clip, q in qs(split):
        if q.get("subcategory") != "future_motion_label": continue
        m = R.search(q["question"])
        if not m: continue
        cam = m.group(1).replace(" ", "_"); x1, y1, x2, y2 = map(int, m.groups()[1:5]); x, y = map(float, m.groups()[5:7])
        mp = f"cache/{split}/{clip}/metadata.json"
        if not os.path.isfile(mp): continue
        if clip not in rigs: rigs[clip] = camgeo.Rig(json.load(open(mp)))
        r = rigs[clip]
        if cam not in r.cams: continue
        us = [r.project(cam, (x, y, z)) for z in (0.3, 0.9, 1.5)]
        us = [u for u, v, ok in us if ok]
        if not us: rows.append((np.nan, False, np.hypot(x, y), x2 - x1)); continue
        u = float(np.mean(us)); w = max(x2 - x1, 1)
        rows.append(((u - (x1 + x2) / 2) / w, x1 - 0.25 * w <= u <= x2 + 0.25 * w, np.hypot(x, y), w))
    a = np.array([r[0] for r in rows], float); ins = np.array([r[1] for r in rows]); d = np.array([r[2] for r in rows])
    fin = np.isfinite(a)
    print(f"{split}: n={len(rows)} projection behind camera {np.mean(~fin):.3f} | inside box (+-25% width) {ins.mean():.3f}"
          f" | |offset| median {np.median(np.abs(a[fin])):.2f} box widths, p90 {np.percentile(np.abs(a[fin]), 90):.2f}")
    for lo, hi in ((0, 15), (15, 30), (30, 200)):
        k = (d >= lo) & (d < hi)
        if k.sum(): print(f"   range {lo}-{hi} m: n={k.sum()} inside {ins[k].mean():.3f}")


# box size vs stated range: for cars, (box height px) * range / focal  ~ car height (m), independent of range.
print("\nimplied object height for 'car' boxes (box height x range / focal), future_motion questions:")
R2 = re.compile(r"For the (\w+) in the (.+?) camera with normalized 2D box \[(\d+), (\d+), (\d+), (\d+)\].*?\[x, y\] = \[(-?[\d.]+), (-?[\d.]+)\]")
for split in ("validation", "test"):
    hs = []; ws = []; rigs = {}
    for clip, q in qs(split):
        if q.get("subcategory") != "future_motion_label": continue
        m = R2.search(q["question"])
        if not m or m.group(1) != "car": continue
        cam = m.group(2).replace(" ", "_"); x1, y1, x2, y2 = map(int, m.groups()[2:6]); x, y = map(float, m.groups()[6:8])
        mp = f"cache/{split}/{clip}/metadata.json"
        if not os.path.isfile(mp): continue
        if clip not in rigs: rigs[clip] = camgeo.Rig(json.load(open(mp)))
        c = rigs[clip].cams.get(cam)
        if c is None or y2 >= 995 or y1 <= 5 or x1 <= 5 or x2 >= 995: continue      # truncated boxes out
        rng = float(np.hypot(x - c["t"][0], y - c["t"][1]))
        hs.append((y2 - y1) / 1000 * c["H"] * rng / c["K"][1, 1]); ws.append(rng)
    hs = np.array(hs); ws = np.array(ws); lh = np.log(hs)
    print(f"  {split}: n={len(hs)} implied height median {np.median(hs):.2f} m, IQR {np.percentile(hs, 25):.2f}-{np.percentile(hs, 75):.2f},"
          f" share outside 1.0-2.5 m {np.mean((hs < 1.0) | (hs > 2.5)):.3f}")
    for lo, hi in ((0, 15), (15, 30), (30, 200)):
        k = (ws >= lo) & (ws < hi)
        if k.sum(): print(f"     range {lo}-{hi} m n={k.sum()} median {np.median(hs[k]):.2f} outside 1.0-2.5 m {np.mean((hs[k] < 1.0) | (hs[k] > 2.5)):.3f}")
