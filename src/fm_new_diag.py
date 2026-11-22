"""2026-11-08: do the new future-motion nets (held-out-year runs, ViT-L and ViT-B) transfer to test better than
net_b16 (p12, -10 answers)? Inputs-only: net-vs-tracker agreement and 'maintaining' share on test vs validation,
validation accuracy by lateral stratum, and how many p19 answers mean(tracker, net) would change. No test labels.
"""
import glob, json, sys, numpy as np
sys.path.insert(0, "src")
import fm_rule as F, camgeo
LAB = ("approaching ego", "moving away from ego", "maintaining relative distance")
rule = lambda c: LAB[0] if c > .5 else (LAB[1] if c < -.5 else LAB[2])
cls = lambda v: np.sign(np.where(np.abs(v) > .5, v, 0))
NETS = ("net_b16", "b16_ho2425", "l16_ho2425")
Q = {s: {json.loads(l)["qid"]: json.loads(l) for l in open(f"artifacts/box3d_q_{s}.jsonl")
         if json.loads(l).get("sub") == "future_motion_label"} for s in ("validation", "test")}
for split, pref in (("validation", "runs_fm/sam21_full"), ("test", "runs_fm/sam21_test")):
    recs = {}
    for f in sorted(glob.glob(f"{pref}_s*of*.json")): recs.update(json.load(open(f)))
    P = {n: json.load(open(f"runs_fm/{n}/pred_{split}.json")) for n in NETS}
    T = F.truth(split, recs) if split == "validation" else {}
    rigs = {}; rows = []
    for q, r in recs.items():
        if any(q not in P[n] for n in NETS) or r.get("missing"): continue
        if r["clip"] not in rigs: rigs[r["clip"]] = camgeo.Rig(json.load(open(f"cache/{split}/{r['clip']}/metadata.json")))
        rows.append((q, F.closing(r, rigs[r["clip"]], "h", edge=3, min_frames=2)[0], abs(r["y"]), T.get(q, np.nan)))
    tr = np.array([x[1] for x in rows]); y = np.array([x[2] for x in rows]); t = np.array([x[3] for x in rows]); ok = ~np.isnan(tr)
    print(f"== {split}: n {len(rows)}, tracker usable {ok.sum()}, in-lane (|y|<1) {(y < 1).sum()}")
    for n in NETS:
        nt = np.array([P[n][x[0]] for x in rows])
        line = (f"  {n:11s} maintaining share {(np.abs(nt) < .5).mean():.2f} | usable: corr {np.corrcoef(nt[ok], tr[ok])[0,1]:.3f} "
                f"agree {(cls(nt[ok]) == cls(tr[ok])).mean():.3f}, in-lane agree {(cls(nt[ok & (y < 1)]) == cls(tr[ok & (y < 1)])).mean():.3f}")
        if T:
            k = ~np.isnan(t); ku = k & ok; kl = ku & (y < 1)
            line += (f" | acc net {(cls(nt[k]) == cls(t[k])).mean():.3f}, usable tracker {(cls(tr[ku]) == cls(t[ku])).mean():.3f} "
                     f"mean {(cls((nt[ku] + tr[ku]) / 2) == cls(t[ku])).mean():.3f}; in-lane n {kl.sum()} tracker "
                     f"{(cls(tr[kl]) == cls(t[kl])).mean():.3f} mean {(cls((nt[kl] + tr[kl]) / 2) == cls(t[kl])).mean():.3f}")
        print(line)
    print(f"  tracker maintaining share (usable) {(np.abs(tr[ok]) < .5).mean():.2f}" + (f", truth {(np.abs(t[~np.isnan(t)]) <= .5).mean():.2f}" if T else ""))

# How many p19 answers would mean(tracker, net) change on test (tracker-usable only; else keep p19's answer)?
recs = {}
for f in sorted(glob.glob("runs_fm/sam21_test_s*of*.json")): recs.update(json.load(open(f)))
p19 = {a["question_id"]: a["answer"] for c in json.load(open("submissions/probes/p19_dist_vitl.json"))["clips"] for a in c["answers"]}
rigs = {}
for n in NETS:
    P = json.load(open(f"runs_fm/{n}/pred_test.json")); ch = 0; usable = 0
    for q, s in Q["test"].items():
        r = recs.get(q)
        if not r or r.get("missing"): continue
        if r["clip"] not in rigs: rigs[r["clip"]] = camgeo.Rig(json.load(open(f"cache/test/{r['clip']}/metadata.json")))
        e = F.closing(r, rigs[r["clip"]], "h", edge=3, min_frames=2)[0]
        if not np.isfinite(e) or q not in P: continue
        usable += 1
        letter = {v: k for k, v in s["choices"].items()}[rule((e + P[q]) / 2)]
        ch += letter != p19[q]
    print(f"test {n}: mean(tracker, net) on {usable} usable changes {ch} p19 answers")
