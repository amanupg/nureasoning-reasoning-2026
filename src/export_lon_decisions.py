"""Per-clip longitudinal driving-decision probabilities for the planning thread (request 2026-10-21).
Source: src/dec_head.py outputs (DINOv3 scene tokens + ego kinematics; trained on TRAIN annotations only;
train = out-of-fold by drive log, validation/test = mean of the 5 fold models) and its kinematics-only control.
Calibrated with one temperature per source fitted on the TRAIN out-of-fold predictions only (no validation labels).
-> artifacts/lon_decision_probs_<split>.json {clip: {"vision": {class: p}, "kinematics": {class: p}}}"""
import json, os
import numpy as np
LON_MAP = {"Gently accelerate (speed up)": "Gently accelerate", "Quickly accelerate (speed up)": "Quickly accelerate"}


def decision(cp):
    fr = json.load(open(os.path.join(cp, "metadata.json")))["frames"]
    for k in (100, 101, 99):
        if k >= len(fr): continue
        r = fr[k].get("reasoning")
        if r and os.path.isfile(os.path.join(cp, r)):
            dr = json.load(open(os.path.join(cp, r))).get("Driving"); d = dr.get("Driving decision") if isinstance(dr, dict) else None
            if isinstance(d, dict) and d.get("Longitudinal"): return LON_MAP.get(d["Longitudinal"], d["Longitudinal"])


def probs(tag, split):
    z = np.load(f"artifacts/{tag}_{split}.npz", allow_pickle=True)
    p = z["clip"][:, :8].astype(np.float64); p = p / p.sum(1, keepdims=True)
    return [str(n) for n in z["names"]], p, [str(c) for c in z["clip_cols"][:8]]


def temp(p, T):
    l = np.log(np.clip(p, 1e-9, 1)) / T; l -= l.max(1, keepdims=True); e = np.exp(l); return e / e.sum(1, keepdims=True)


def nll(p, y): return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1))))


out = {s: {} for s in ("train", "validation", "test")}
for tag, key in (("decpred", "vision"), ("decpred_novision", "kinematics")):
    names, p, cols = probs(tag, "train")
    lab = [(i, cols.index(d)) for i, n in enumerate(names) if os.path.isdir(f"cache/train/{n}") and (d := decision(f"cache/train/{n}")) in cols]
    ii = np.array([i for i, _ in lab]); y = np.array([c for _, c in lab])
    Ts = np.linspace(0.5, 4, 71); T = float(Ts[np.argmin([nll(temp(p[ii], t), y) for t in Ts])])
    print(f"{key}: temperature {T:.2f} (train OOF nll {nll(p[ii], y):.3f} -> {nll(temp(p[ii], T), y):.3f})")
    for split in out:
        names, p, cols = probs(tag, split); q = temp(p, T)
        for n, r in zip(names, q): out[split].setdefault(n, {})[key] = {c: round(float(v), 5) for c, v in zip(cols, r)}
        if split == "validation":
            lab = [(i, cols.index(d)) for i, n in enumerate(names) if (d := decision(f"cache/validation/{n}")) in cols]
            ii = np.array([i for i, _ in lab]); y = np.array([c for _, c in lab])
            print(f"   validation (never trained on): n={len(y)} acc {np.mean(q[ii].argmax(1) == y):.3f} nll raw {nll(p[ii], y):.3f} -> calibrated {nll(q[ii], y):.3f}")
for split, d in out.items():
    json.dump(d, open(f"artifacts/lon_decision_probs_{split}.json", "w"))
    print(split, len(d), "clips")
