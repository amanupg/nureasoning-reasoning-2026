"""Optimise arc-length prediction against the REAL objective, not MSE.

Per-clip planning reward (gates excluded, they are invisible on test):
    0.3 * progress + 0.2 * comfort + 0.5 * human
with progress = clip(pred_arc / gt_arc, 0, 1)  [=1 when gt_arc < 0.1]
     human    = smoothstep on FDE = hypot(arc_err, lateral_offset)

Progress is clamped above, so OVERSHOOT costs only FDE while UNDERSHOOT costs
FDE and progress. The optimiser is therefore expected to pick a positive bias.
"""
import json, numpy as np, itertools, sys

R = [r for r in json.load(open("runs/arc_features.json")) if r["has_route"]]
v0 = np.array([r["v0"] for r in R]); a0 = np.array([r["a0"] for r in R])
vh1 = np.array([r["v_h1"] for r in R]); ach = np.array([r["acc_h"] for r in R])
arc = np.array([r["arc"] for r in R]); lat = np.array([r["lat"] for r in R])


def human(fde):
    f, z = 1.0, 8.0
    t = np.clip((fde - f) / (z - f), 0, 1)
    return np.where(fde <= f, 1.0, np.where(fde >= z, 0.0, 1 - t * t * (3 - 2 * t)))


def progress(pred):
    p = np.where(arc < 0.1, 1.0, np.clip(pred / np.where(arc < 0.1, 1.0, arc), 0, 1))
    return p


def reward(pred):
    pred = np.maximum(pred, 0.0)
    fde = np.hypot(arc - pred, lat)
    h = human(fde)
    return 0.3 * progress(pred) + 0.2 * 1.0 + 0.5 * h, h, progress(pred)


print("=== grid search: pred = v0*5 + 0.5*damp*a0*25 + bias ===")
best = None
for damp in np.arange(0.0, 1.55, 0.05):
    for bias in np.arange(-2.0, 6.01, 0.25):
        pred = v0 * 5 + 0.5 * damp * a0 * 25 + bias
        r, h, p = reward(pred)
        if best is None or r.mean() > best[0]:
            best = (r.mean(), damp, bias, h.mean(), p.mean())
print(f"  best reward={best[0]:.4f}  damp={best[1]:.2f}  bias={best[2]:+.2f}  "
      f"human={best[3]:.4f}  progress={best[4]:.4f}")

base_r, base_h, base_p = reward(v0 * 5)
print(f"  constant velocity           reward={base_r.mean():.4f}  "
      f"human={base_h.mean():.4f}  progress={base_p.mean():.4f}")

print("\n=== is the optimal bias really positive? (damp fixed at best) ===")
for bias in (-1, 0, 1, 2, 3, 4):
    pred = v0 * 5 + 0.5 * best[1] * a0 * 25 + bias
    r, h, p = reward(pred)
    print(f"  bias {bias:+d} m : reward={r.mean():.4f}  human={h.mean():.4f}  "
          f"progress={p.mean():.4f}")

print("\n=== add a stopped-ego rule (v0 < thresh -> predict 0) ===")
damp, bias = best[1], best[2]
base = v0 * 5 + 0.5 * damp * a0 * 25 + bias
for th in (0.0, 0.3, 0.5, 1.0, 1.5, 2.0):
    pred = np.where(v0 < th, 0.0, base)
    r, _, _ = reward(pred)
    print(f"  v0 < {th:.1f} m/s -> 0 : reward={r.mean():.4f}")

print("\n=== oracle / bounds ===")
r_o, h_o, p_o = reward(arc)
print(f"  ORACLE arc length : reward={r_o.mean():.4f} human={h_o.mean():.4f}")
