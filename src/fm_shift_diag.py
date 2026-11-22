"""Why did the future-motion net (runs_fm/net_b16) not transfer to test (p12 -10 answers)? Inputs-only diagnostic:
agreement of the net's closing speed with the SAM 2.1 tracker's, validation vs test; and on validation, accuracy
by the same strata. No test labels used (none exist locally).
"""
import glob, json, math, sys, numpy as np
sys.path.insert(0, "src")
import fm_rule as F, camgeo
for split, pref in (("validation", "runs_fm/sam21_full"), ("test", "runs_fm/sam21_test")):
    recs = {}
    for f in sorted(glob.glob(f"{pref}_s*of*.json")): recs.update(json.load(open(f)))
    net = json.load(open(f"runs_fm/net_b16/pred_{split}.json")); rigs = {}
    T = F.truth(split, recs) if split == "validation" else {}
    rows = []
    for q, r in recs.items():
        if q not in net or r.get("missing"): continue
        if r["clip"] not in rigs: rigs[r["clip"]] = camgeo.Rig(json.load(open(f"cache/{split}/{r['clip']}/metadata.json")))
        tr = F.closing(r, rigs[r["clip"]], "h", edge=3, min_frames=2)[0]
        rows.append((q, tr, net[q], math.hypot(r["x"], r["y"]), r["cam"], T.get(q)))
    tr = np.array([x[1] for x in rows]); nt = np.array([x[2] for x in rows]); rng = np.array([x[3] for x in rows])
    ok = ~np.isnan(tr); cls = lambda v: np.sign(np.where(np.abs(v) > .5, v, 0))
    print(f"== {split}: n {len(rows)}, tracker usable {ok.mean():.2f}")
    print(f"  net closing: median {np.median(nt):+.2f}, |c|<0.5 share {(np.abs(nt) < .5).mean():.2f}, approaching {(nt > .5).mean():.2f}, away {(nt < -.5).mean():.2f}")
    print(f"  tracker closing: median {np.nanmedian(tr):+.2f}, approaching {(tr[ok] > .5).mean():.2f}, away {(tr[ok] < -.5).mean():.2f}")
    print(f"  usable: corr(net, tracker) {np.corrcoef(nt[ok], tr[ok])[0,1]:.3f}, class agreement {(cls(nt[ok]) == cls(tr[ok])).mean():.3f}")
    for lo, hi in ((0, 20), (20, 40), (40, 200)):
        m = ok & (rng >= lo) & (rng < hi)
        line = f"  range {lo}-{hi} m n {m.sum():4d}: agree {(cls(nt[m]) == cls(tr[m])).mean():.3f}"
        if T:
            t = np.array([x[5] if x[5] is not None else np.nan for x in rows]); mm = m & ~np.isnan(t)
            line += f" | acc net {(cls(nt[mm]) == cls(t[mm])).mean():.3f} tracker {(cls(tr[mm]) == cls(t[mm])).mean():.3f}"
        print(line)
    cams = sorted(set(x[4] for x in rows))
    print("  cams:", {c: round(float(np.mean([x[4] == c for x in rows])), 2) for c in cams})

# --- 2026-11-08: does the duplicated key frame (test t100 == t99, so the frame-90 -> 100 pair spans 0.9 s) explain it?
print("== test, by key-frame duplication")
recs = {}
for f in sorted(glob.glob("runs_fm/sam21_test_s*of*.json")): recs.update(json.load(open(f)))
net = json.load(open("runs_fm/net_b16/pred_test.json")); rigs = {}; dupc = {}
R = []
for q, r in recs.items():
    if q not in net or r.get("missing"): continue
    c = r["clip"]
    if c not in rigs:
        md = json.load(open(f"cache/test/{c}/metadata.json")); rigs[c] = camgeo.Rig(md)
        fr = {f["frame_index"]: f for f in md["frames"]}; dupc[c] = fr[100]["timestamp_us"] == fr[99]["timestamp_us"]
    R.append((F.closing(r, rigs[c], "h", edge=3, min_frames=2)[0], net[q], dupc[c]))
tr = np.array([x[0] for x in R]); nt = np.array([x[1] for x in R]); du = np.array([x[2] for x in R]); ok = ~np.isnan(tr)
cls = lambda v: np.sign(np.where(np.abs(v) > .5, v, 0))
for nm, m in (("duplicated", du), ("not duplicated", ~du)):
    mm = m & ok
    print(f"  {nm:15s} n {m.sum():3d} usable {mm.sum():3d}: agree {(cls(nt[mm]) == cls(tr[mm])).mean():.3f}, corr {np.corrcoef(nt[mm], tr[mm])[0,1]:.3f}, "
          f"net |c|<0.5 {(np.abs(nt[m]) < .5).mean():.2f}, agree with net x 1/0.9 {(cls(nt[mm] / 0.9) == cls(tr[mm])).mean():.3f}")

# --- 2026-11-08: test fm objects are laterally centred (|y| median 0.1 m vs 2.4 m on validation). Validation by |y| stratum.
print("== validation by lateral offset (truth = annotated closing speed rule)")
recs = {}
for f in sorted(glob.glob("runs_fm/sam21_full_s*of*.json")): recs.update(json.load(open(f)))
net = json.load(open("runs_fm/net_b16/pred_validation.json")); T = F.truth("validation", recs); rigs = {}
rows = []
for q, r in recs.items():
    if q not in net or q not in T or r.get("missing"): continue
    if r["clip"] not in rigs: rigs[r["clip"]] = camgeo.Rig(json.load(open(f"cache/validation/{r['clip']}/metadata.json")))
    rows.append((abs(r["y"]), F.closing(r, rigs[r["clip"]], "h", edge=3, min_frames=2)[0], net[q], T[q]))
y = np.array([x[0] for x in rows]); tr = np.array([x[1] for x in rows]); nt = np.array([x[2] for x in rows]); t = np.array([x[3] for x in rows])
cls = lambda v: np.sign(np.where(np.abs(v) > .5, v, 0)); ok = ~np.isnan(tr)
for lo, hi in ((0, 1), (1, 2.5), (2.5, 99)):
    m = (y >= lo) & (y < hi); mm = m & ok
    print(f"  |y| {lo}-{hi} m n {m.sum():4d}: truth maintaining {(np.abs(t[m]) <= .5).mean():.2f} | net acc {(cls(nt[m]) == cls(t[m])).mean():.3f} "
          f"| usable {mm.sum()}: net {(cls(nt[mm]) == cls(t[mm])).mean():.3f} tracker {(cls(tr[mm]) == cls(t[mm])).mean():.3f} "
          f"mean {(cls((nt[mm] + tr[mm]) / 2) == cls(t[mm])).mean():.3f} | net-tracker agree {(cls(nt[mm]) == cls(tr[mm])).mean():.3f}")
