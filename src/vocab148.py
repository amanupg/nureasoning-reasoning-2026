"""The 148-candidate trajectory vocabulary: union of
  REL: arc = scale x kinematic estimate (9 scales) x lateral offset (7)  -> 63
  ABS: absolute arc 0..77 m (17)                  x lateral offset (5)  -> 85
Measured oracle on 257 validation route clips: 63-only 0.731, union 0.835
(stopped clips 0.480 -> 0.947: relative arcs cannot express "pull away").
"""
import numpy as np
import planners
from vocab_labels import ARC_SCALES, LAT_OFFSETS, build_candidate

ABS_ARCS = [0, 1, 2.5, 5, 7.5, 10, 13, 16, 20, 25, 30, 36, 43, 50, 58, 67, 77]
ABS_LATS = [-1.5, -0.75, 0.0, 0.75, 1.5]
VOCAB148 = ([("rel", s, l) for s in ARC_SCALES for l in LAT_OFFSETS]
            + [("abs", L, l) for L in ABS_ARCS for l in ABS_LATS])
DEFAULT_INDEX = VOCAB148.index(("rel", 1.0, 0.0))     # what we submit today


def _lateral(t, lat):
    if lat and len(t) > 3:
        d = t[3, :2] - t[0, :2]; n = float(np.hypot(*d))
        if n > 1e-6:
            nrm = np.array([-d[1] / n, d[0] / n]); w = np.clip(np.arange(51) * 0.1 / 1.5, 0, 1)[:, None]
            t = t.copy(); t[:, :2] += nrm * lat * w
    return t


def build(meta, fr, ki, ego, route, spec):
    kind, v, lat = spec
    if kind == "rel":
        L0 = planners.arc_length_kinematic(planners._speed(ego), planners._lon_accel(ego))
        return build_candidate(meta, fr, ki, ego, route, v, lat, L0)
    K0 = planners.arc_length_kinematic(planners._speed(ego), planners._lon_accel(ego), 0.75, 0.0)
    t = planners.route_follow(None, meta, fr, ki, ego, route, damp=0.75, bias=v - K0, hold_lat=1.0)
    return _lateral(t, lat)


def spec_vec(spec):
    """Numeric encoding of a vocabulary entry for the scorer's query."""
    kind, v, lat = spec
    return [1.0 if kind == "abs" else 0.0, v / 50.0 if kind == "abs" else v, lat]
