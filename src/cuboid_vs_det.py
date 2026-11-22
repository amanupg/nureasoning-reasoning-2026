"""Hypothesis check (inputs + validation annotations only): are test question boxes projected 3D cuboids rather than
2D detector boxes? On validation, project each detection's corners_3d_ego into its camera and measure IoU with the
generator's detection_bbox_2d (pixels). Compare with the test-box-vs-SAM-3 IoU (runs_vis). Usage: python src/cuboid_vs_det.py"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camgeo


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def cuboid_box(rig, cam, corners):
    c = rig.cams[cam]; P = []
    for p in corners:
        pc = c["R"].T @ (np.asarray(p, float) - c["t"])
        if pc[2] <= 0.5: return None
        uvw = c["K"] @ pc; P.append((uvw[0] / uvw[2], uvw[1] / uvw[2]))
    P = np.array(P); x1, y1 = P.min(0); x2, y2 = P.max(0)
    return [max(0, x1), max(0, y1), min(c["W"], x2), min(c["H"], y2)]


def main():
    res = {}
    for d in sorted(glob.glob("cache/validation/*"))[:400]:
        md = json.load(open(f"{d}/metadata.json")); rig = camgeo.Rig(md)
        fr = [f for f in md["frames"] if f["frame_index"] == 100]
        if not fr or not fr[0].get("reasoning"): continue
        try: r = json.load(open(f"{d}/{fr[0]['reasoning']}"))
        except Exception: continue
        for cam, pc in ((r.get("Spatial") or {}).get("per_camera_results") or {}).items():
            if cam not in rig.cams: continue
            for o in pc.get("objects") or []:
                b = o.get("detection_bbox_2d"); c3 = (o.get("detection_bbox_3d") or {}).get("corners_3d_ego")
                if not b or not c3: continue
                cb = cuboid_box(rig, cam, c3)
                if cb is None or cb[2] <= cb[0] or cb[3] <= cb[1]: continue
                cat = o.get("detection_label", "?")
                w_det = b[2] - b[0]; w_cub = cb[2] - cb[0]
                res.setdefault(cat, []).append((iou(b, cb), w_cub / max(w_det, 1), (cb[3] - cb[1]) / max(b[3] - b[1], 1)))
    for cat, v in sorted(res.items(), key=lambda x: -len(x[1])):
        v = np.array(v)
        if len(v) < 30: continue
        print(f"{cat:14s} n={len(v):5d} IoU(det, cuboid) q25/50/75 {np.round(np.percentile(v[:, 0], [25, 50, 75]), 3)} "
              f"width ratio cub/det median {np.median(v[:, 1]):.2f} height ratio {np.median(v[:, 2]):.2f}")


if __name__ == "__main__":
    main()
