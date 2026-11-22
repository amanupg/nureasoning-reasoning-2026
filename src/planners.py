"""Trajectory providers. All return GLOBAL-frame (51,3) [x, y, yaw].

Design follows the measured structure of NPS:
  * human-likeness (w=0.5) reads ONLY the endpoint, and the human endpoint sits
    on the route (median lateral offset 0.39 m). So we place the endpoint on the
    route and spend our effort on ONE number: arc length travelled in 5 s.
  * progress (w=0.3) is measured by projection onto that same route, so a
    route-following path maximises it for a given arc length.
  * comfort (w=0.2) only sees 11 subsampled points through a global quadratic
    fit, so a smooth speed profile passes trivially.
  * the two gates are invisible on test (no map, no annotations); hugging the
    route is our best available proxy for staying drivable.
"""
from __future__ import annotations

import math
import numpy as np
from shapely.geometry import Point

STEPS, DT, HORIZON = 51, 0.1, 5.0
T = np.arange(STEPS) * DT

# comfort bounds from BenchmarkConfig, with margin
MAX_ACC, MIN_ACC = 2.40, -4.05
MAX_LAT_ACC = 4.89


def _speed(ego):
    v = ego.velocity if isinstance(ego.velocity, dict) else {}
    return float(math.hypot(v.get("vx", 0.0), v.get("vy", 0.0)))


def _lon_accel(ego):
    a = ego.acceleration if isinstance(ego.acceleration, dict) else {}
    return float(a.get("ax", 0.0))


def arc_length_kinematic(v0: float, a0: float, damp: float = 0.75,
                         bias: float = -1.25) -> float:
    """Predicted distance travelled along the route in 5 s.

    Grid-searched on 849 validation clips against the REAL objective
    (0.3*progress + 0.5*human), not MSE: least squares scores WORSE than this
    because the human term is a flat-topped smoothstep, so squared error is
    mis-specified.

    The optimal bias is NEGATIVE (-1.25 m). v0*5 systematically overshoots,
    because nuReasoning selects decision-critical long-tail clips in which the
    ego decelerates more often than in ordinary driving.
    """
    a = float(np.clip(a0 * damp, MIN_ACC, MAX_ACC))
    t_stop = HORIZON if a >= 0 else min(HORIZON, max(0.0, -v0 / a))
    s = v0 * t_stop + 0.5 * a * t_stop ** 2 + bias
    return float(max(0.0, s))


def _profile(v0: float, L: float) -> np.ndarray:
    """Monotone arc-length samples s(t) reaching L at t=5, starting at speed v0."""
    a = 2.0 * (L - v0 * HORIZON) / (HORIZON ** 2)
    a = float(np.clip(a, MIN_ACC, MAX_ACC))
    s = v0 * T + 0.5 * a * T ** 2
    s = np.maximum.accumulate(np.clip(s, 0.0, None))      # never reverse
    if s[-1] > 1e-6:                                      # rescale to hit L exactly
        s = s * (L / s[-1])
    else:
        s = np.linspace(0.0, L, STEPS)
    return s


def route_follow(clip_path, meta, frames, ki, ego, route_line, damp=0.75, bias=-1.25,
                 curvature_limit=False, hold_lat=1.0, lat_decay_s=1.5):
    """Place the endpoint on the route at the predicted arc length."""
    x, y, yaw = ego.pose["x"], ego.pose["y"], ego.pose["yaw"]
    v0, a0 = _speed(ego), _lon_accel(ego)

    if route_line is None or route_line.is_empty or route_line.length < 1.0:
        return constant_turn(clip_path, meta, frames, ki, ego, None)

    s0 = float(route_line.project(Point(x, y)))
    L = arc_length_kinematic(v0, a0, damp, bias)
    L = max(0.0, L)
    # NOTE: a lateral-acceleration curvature limiter was implemented and
    # measured here. It bound on 0/277 route clips even after fixing a
    # double-division bug: route geometry never violates the 4.89 m/s^2 bound
    # at observed speeds. Removed rather than kept as dead code.
    L = min(L, max(route_line.length - s0, 0.0))

    s = s0 + _profile(v0, L)
    s = np.clip(s, 0.0, route_line.length)
    pts = np.array([[route_line.interpolate(float(si)).x,
                     route_line.interpolate(float(si)).y] for si in s])

    # Start exactly at the ego pose. The LONGITUDINAL part of the offset is an
    # artefact of projection and is blended out quickly; the LATERAL part is a
    # real driving choice (the human does not sit on the map centerline, and in
    # work zones the centerline runs through the cone line). hold_lat controls
    # how much of that lateral offset we keep at t=5 s.
    off = np.array([x, y]) - pts[0]
    d0 = pts[min(3, len(pts) - 1)] - pts[0]
    n0 = float(np.hypot(*d0))
    if n0 > 1e-6:
        tang = d0 / n0
        nrm = np.array([-tang[1], tang[0]])
        lon = float(off @ tang) * tang
        lat = float(off @ nrm) * nrm
    else:
        lon, lat = off, np.zeros(2)
    w_lon = np.clip(1.0 - T / 1.5, 0.0, 1.0)[:, None]
    w_lat = (hold_lat + (1.0 - hold_lat) * np.clip(1.0 - T / lat_decay_s, 0.0, 1.0))[:, None]
    pts = pts + lon * w_lon + lat * w_lat

    d = np.diff(pts, axis=0)
    head = np.arctan2(d[:, 1], d[:, 0])
    moving = np.hypot(d[:, 0], d[:, 1]) > 1e-3
    if moving.any():
        head = np.where(moving, head, np.nan)
        idx = np.arange(len(head))
        good = ~np.isnan(head)
        head = np.interp(idx, idx[good], head[good])
        yaws = np.concatenate([[yaw], head])
        yaws[0] = yaw
    else:
        yaws = np.full(STEPS, yaw)
    return np.column_stack([pts[:, 0], pts[:, 1], yaws])


def constant_turn(clip_path, meta, frames, ki, ego, route_line=None):
    """Fallback with no route: extrapolate speed and yaw-rate from 3 s of history."""
    x, y, yaw = ego.pose["x"], ego.pose["y"], ego.pose["yaw"]
    v0, a0 = _speed(ego), _lon_accel(ego)
    hist = getattr(ego, "trajectory_history", None) or []
    omega = 0.0
    if len(hist) >= 6:
        h = np.array(hist, float)
        yaws = np.unwrap(h[:, 2])
        k = min(10, len(yaws) - 1)
        omega = float((yaws[-1] - yaws[-1 - k]) / (k * DT))
    omega = float(np.clip(omega, -0.4, 0.4))
    L = arc_length_kinematic(v0, a0)
    s = _profile(v0, L)
    th = yaw + omega * T
    ds = np.diff(s, prepend=0.0)
    xs = x + np.cumsum(ds * np.cos(th))
    ys = y + np.cumsum(ds * np.sin(th))
    return np.column_stack([xs, ys, th])


# --------------------------------------------------------------------------
# Learned arc length (decision-theoretic)
# --------------------------------------------------------------------------
_ARC = None


def _load_arc(path="artifacts/arc_bayes.pkl"):
    global _ARC
    if _ARC is None:
        import pickle
        with open(path, "rb") as fh:
            _ARC = pickle.load(fh)
    return _ARC


def route_bayes(clip_path, meta, frames, ki, ego, route_line):
    """route_follow, but the arc length comes from the Bayes-optimal action
    under the real planning reward rather than a kinematic rule."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from arc_data import features_from
    from arc_bayes import sample_matrix, bayes_action

    if route_line is None or route_line.is_empty or route_line.length < 1.0:
        return constant_turn(clip_path, meta, frames, ki, ego, None)

    b = _load_arc()
    f = features_from(meta, frames, ki, ego)
    X = np.array([[f.get(k, np.nan) for k in b["feats"]]], float)
    S = sample_matrix(b["quantile_models"], X)
    L = float(bayes_action(S, b["lat_hat"])[0])
    return _place_on_route(ego, route_line, L)


def _place_on_route(ego, route_line, L):
    x, y, yaw = ego.pose["x"], ego.pose["y"], ego.pose["yaw"]
    v0 = _speed(ego)
    s0 = float(route_line.project(Point(x, y)))
    L = float(np.clip(L, 0.0, max(route_line.length - s0, 0.0)))
    s = np.clip(s0 + _profile(v0, L), 0.0, route_line.length)
    pts = np.array([[route_line.interpolate(float(si)).x,
                     route_line.interpolate(float(si)).y] for si in s])
    off = np.array([x, y]) - pts[0]
    w = np.clip(1.0 - T / 1.5, 0.0, 1.0)[:, None]
    pts = pts + off * w
    d = np.diff(pts, axis=0)
    head = np.arctan2(d[:, 1], d[:, 0])
    moving = np.hypot(d[:, 0], d[:, 1]) > 1e-3
    if moving.any():
        head = np.where(moving, head, np.nan)
        idx = np.arange(len(head)); good = ~np.isnan(head)
        head = np.interp(idx, idx[good], head[good])
        yaws = np.concatenate([[yaw], head]); yaws[0] = yaw
    else:
        yaws = np.full(STEPS, yaw)
    return np.column_stack([pts[:, 0], pts[:, 1], yaws])
