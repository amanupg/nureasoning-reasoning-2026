"""Generate LABELLED VQA for cached clips using the devkit's own generator.

The generator opens camera JPEGs only to read their pixel dimensions, which our
image-free cache does not have. Every camera in the release is 2816x1856 and the
size is also recorded in metadata.json camera_calibrations, so we patch the size
lookup instead of fabricating ~24 GB of stub images.

Output is the same schema as the released test reasoning_questions.json, but WITH
the `answer` field, giving us a local reasoning harness on the challenge's own
question distribution.
"""
import os, sys, json, random, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "probe"))
import _stub_torch  # noqa: F401
sys.path.insert(0, os.path.join(ROOT, "recon", "devkit"))

import nureasoning.vqa.generate as G

CAM_WH = (2816, 1856)


def _patched_image_size(image_path):
    return CAM_WH


def verify_uniform(root, n=50):
    """Check the assumption before relying on it."""
    seen = set()
    for d in sorted(os.listdir(root))[:n]:
        p = os.path.join(root, d, "metadata.json")
        if not os.path.isfile(p):
            continue
        for cam, c in (json.load(open(p)).get("camera_calibrations") or {}).items():
            seen.add((c.get("width"), c.get("height")))
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="cache/validation")
    ap.add_argument("--output", default="vqa_val")
    ap.add_argument("--key-frame", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    seen = verify_uniform(a.data_root)
    if seen != {CAM_WH}:
        raise SystemExit(f"camera sizes are not uniformly {CAM_WH}: {seen}")
    print(f"[gen_vqa] verified all camera calibrations are {CAM_WH[0]}x{CAM_WH[1]}")

    G._image_size = _patched_image_size
    if hasattr(G, "_normalized_bbox_coordinates"):
        pass  # it calls _image_size by global name, so the patch applies

    random.seed(a.seed)
    os.makedirs(a.output, exist_ok=True)
    clips = sorted(os.path.join(a.data_root, d) for d in os.listdir(a.data_root)
                   if os.path.isfile(os.path.join(a.data_root, d, "metadata.json")))
    if a.limit:
        clips = clips[:a.limit]

    ok = fail = 0
    for i, cp in enumerate(clips):
        try:
            G.process_clip(cp, a.output, key_frame_index=a.key_frame,
                           history_frames=3, use_gemini=False, api_key=None)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"  FAIL {os.path.basename(cp)}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(clips)}] ok={ok} fail={fail}", flush=True)
    print(f"[gen_vqa] done ok={ok} fail={fail} -> {a.output}")


if __name__ == "__main__":
    main()
