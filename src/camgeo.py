"""Camera geometry from each clip's own calibration (generator-independent).

Conventions copied from the devkit's planning/visualize.py: the lidar frame is
treated as the ego frame, sensor2lidar_rotation is a [w, x, y, z] quaternion,
and projection is pinhole with depth > 0.5 m. Image coordinates are returned
normalised to 0..1000, the scale the questions use.
"""
import json, math
import numpy as np

NAME2KEY = {"front": "CAM_M_F", "front_left": "CAM_M_L0", "front_right": "CAM_M_R0",
            "left": "CAM_M_L1", "right": "CAM_M_R1", "back": "CAM_M_B",
            "back_left": "CAM_M_L2", "back_right": "CAM_M_R2"}


def _quat(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Rig:
    def __init__(self, metadata):
        self.cams = {}
        for name, key in NAME2KEY.items():
            c = (metadata.get("camera_calibrations") or {}).get(key)
            if not c: continue
            self.cams[name] = dict(K=np.array(c["intrinsic"], float), R=_quat(*c["sensor2lidar_rotation"]),
                                   t=np.array(c["sensor2lidar_translation"], float),
                                   W=float(c["width"]), H=float(c["height"]))

    def project(self, name, p):
        """Ego-frame point (x, y, z) -> (u, v) in 0..1000 and in-front flag."""
        c = self.cams[name]
        pc = c["R"].T @ (np.asarray(p, float) - c["t"])
        if pc[2] <= 0.5:
            return None, None, False
        uvw = c["K"] @ pc
        return 1000 * uvw[0] / uvw[2] / c["W"], 1000 * uvw[1] / uvw[2] / c["H"], True

    def ray(self, name, u, v):
        """Pixel (normalised 0..1000) -> unit ray direction in the ego frame, and camera centre."""
        c = self.cams[name]
        px = np.array([u / 1000 * c["W"], v / 1000 * c["H"], 1.0])
        d = c["R"] @ (np.linalg.inv(c["K"]) @ px)
        return d / np.linalg.norm(d), c["t"]

    def axis_angle(self, name, x, y):
        """Horizontal angle (deg) between the camera optical axis and the bearing to (x, y)."""
        c = self.cams[name]
        ax = c["R"] @ np.array([0, 0, 1.0])
        a1 = math.atan2(ax[1], ax[0]); a2 = math.atan2(y - c["t"][1], x - c["t"][0])
        return abs((math.degrees(a2 - a1) + 180) % 360 - 180)


# ------------------------------------------------------------------ answers
def region_name(u, v):
    return f'{"upper" if v < 333 else "middle" if v < 667 else "lower"} {"left" if u < 333 else "center" if u < 667 else "right"}'


def in_image(u, v, ok, margin=0.0):
    return ok and -margin <= u <= 1000 + margin and -margin <= v <= 1000 + margin


def best_camera(rig, x, y, z, names):
    """Camera that sees the point most centrally: among cameras whose image
    contains the projected point, the one with the smallest angle between its
    optical axis and the bearing to the point. Falls back to smallest angle."""
    inside = [n for n in names if n in rig.cams and in_image(*rig.project(n, (x, y, z)))]
    pool = inside or [n for n in names if n in rig.cams]
    return min(pool, key=lambda n: rig.axis_angle(n, x, y)) if pool else None


def cameras_seeing(rig, x, y, z, margin=0.0):
    return sorted(n for n in rig.cams if in_image(*rig.project(n, (x, y, z)), margin))


def best_pair(rig, x, y, z, choice_texts):
    """Camera pair observing the point, formatted like the generator (sorted names,
    first two). If fewer than two cameras contain the point, use the two cameras
    whose optical axes are closest to its bearing."""
    cs = cameras_seeing(rig, x, y, z)
    if len(cs) < 2:
        cs = sorted(sorted(rig.cams, key=lambda n: rig.axis_angle(n, x, y))[:2])
    pair = " + ".join(cs[:2])
    norm = lambda s: " ".join(s.lower().replace("_", " ").split())
    return pair if norm(pair) in {norm(t) for t in choice_texts} else None
