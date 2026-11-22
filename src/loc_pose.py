"""Global ego pose (UTM x, y, yaw) at the key frame (100) per clip -> artifacts/loc_pose_{split}.json.
Inputs only (test has these poses). Used to test a location speed prior for planning.
  .venv/bin/python src/loc_pose.py [splits]
"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "recon", "devkit"))


def one(args):
    from nureasoning.common.pickle_io import load_pickle
    split, clip = args
    try:
        md = json.load(open(os.path.join(ROOT, "cache", split, clip, "metadata.json")))
        fr = {f["frame_index"]: f for f in md["frames"]}
        e = load_pickle(os.path.join(ROOT, "cache", split, clip, fr[100]["ego_state"]))
        return clip, dict(x=e.pose["x"], y=e.pose["y"], yaw=e.pose["yaw"], loc=md.get("clip_location"), log=md.get("log_name"))
    except Exception:
        return clip, None


if __name__ == "__main__":
    for split in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["train", "validation", "test"]):
        clips = sorted(os.listdir(os.path.join(ROOT, "cache", split)))
        with ProcessPoolExecutor(8) as ex:
            R = {c: o for c, o in ex.map(one, [(split, c) for c in clips], chunksize=32) if o}
        json.dump(R, open(os.path.join(ROOT, "artifacts", f"loc_pose_{split}.json"), "w"))
        print(split, len(R), "/", len(clips), flush=True)
