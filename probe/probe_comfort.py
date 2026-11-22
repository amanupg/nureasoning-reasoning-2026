import probe._stub_torch  # noqa: F401  (must precede the devkit import)
"""Probe the real devkit scorer: what does comfort actually constrain?

Verifies two claims before we build anything on them:
  1. human-likeness reads only the final waypoint (FDE), not the path shape.
  2. comfort subsamples to 11 points and fits a global low-order polynomial,
     so it barely constrains the trajectory at all.
"""
import sys, numpy as np
sys.path.insert(0, "recon/devkit")
from nureasoning.planning.benchmark import (
    BenchmarkConfig, score_comfort, score_human_likeness, _approx_derivative,
)

cfg = BenchmarkConfig()
t = np.arange(51) * 0.1

def traj(x, y, yaw=None):
    return np.stack([x, y, np.zeros(51) if yaw is None else yaw], axis=1)

print("=== CLAIM 1: human-likeness depends only on the LAST waypoint ===")
gt = traj(10.0 * t, np.zeros(51))                      # GT: straight, 10 m/s
smooth = traj(10.0 * t, np.zeros(51))
# same endpoint, wildly different path in between
wild = traj(10.0 * t + 6.0 * np.sin(np.pi * t / 5.0), 4.0 * np.sin(np.pi * t / 5.0))
for name, p in [("identical", smooth), ("wild detour, same endpoint", wild)]:
    s, d = score_human_likeness(p, gt, cfg)
    print(f"  {name:28} human={s:.4f}  ADE={d['ade']:6.3f}  FDE={d['fde']:.4f}")

print("\n=== CLAIM 2: comfort subsamples to 11 pts and fits a global polynomial ===")
sub = np.arange(0, 51, 5)
print(f"  comfort stride=5 -> evaluates indices {list(sub)} ({len(sub)} points)")
# Savitzky-Golay window = min(15, 11) = 11 == n  -> one global fit over 0..5s
x = 10.0 * t[sub]
acc = _approx_derivative(x, 0.5, deriv_order=2, poly_order=2, window_length=15)
print(f"  2nd-derivative of a straight constant-velocity ramp: {np.round(acc,9)}")
x2 = 10.0 * t[sub] + 0.5 * 1.5 * t[sub] ** 2
acc2 = _approx_derivative(x2, 0.5, deriv_order=2, poly_order=2, window_length=15)
print(f"  2nd-derivative of a true 1.5 m/s^2 ramp:  {np.round(acc2,6)}  (constant => global quadratic fit)")

print("\n=== CLAIM 2b: comfort ignores everything between the 11 sampled points ===")
base_x = 10.0 * t
s_base, _ = score_comfort(traj(base_x, np.zeros(51)), cfg)
# inject a violent zig-zag that is exactly zero at every 5th index
spike = np.zeros(51); spike[np.arange(51) % 5 != 0] = 3.0
s_spike, d_spike = score_comfort(traj(base_x, spike), cfg)
print(f"  clean constant-velocity            comfort={s_base}")
print(f"  +3 m lateral zig-zag between pts   comfort={s_spike}   (true lat accel would be ~1200 m/s^2)")
print(f"     reported max_abs_lat_accel={d_spike['max_abs_lat_accel']:.4f}")

print("\n=== CLAIM 2c: how hard is comfort to pass? sweep constant accel / yaw rate ===")
for a in [0.0, 1.0, 2.0, 2.3, 2.5, 3.0, -4.0, -4.1, -5.0]:
    x = np.clip(10.0 + a * t, 0, None) * 0  # placeholder
    x = 10.0 * t + 0.5 * a * t ** 2
    s, d = score_comfort(traj(x, np.zeros(51)), cfg)
    print(f"  const lon accel {a:>5.1f} m/s^2 -> comfort={s:.0f}  (measured {d['min_lon_accel']:.2f}..{d['max_lon_accel']:.2f})")
for w in [0.0, 0.2, 0.5, 0.9, 1.0]:
    yaw = w * t
    x = np.cumsum(np.r_[0, np.diff(t)] * 10.0 * np.cos(yaw))
    y = np.cumsum(np.r_[0, np.diff(t)] * 10.0 * np.sin(yaw))
    s, d = score_comfort(traj(x, y, yaw), cfg)
    print(f"  const yaw rate  {w:>5.2f} rad/s -> comfort={s:.0f}  (measured yaw_rate {d['max_abs_yaw_rate']:.3f}, lat_acc {d['max_abs_lat_accel']:.2f})")
