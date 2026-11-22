"""Alpamayo 1.5 chain-of-causation text -> clip features (fraction of samples mentioning
each cue). A decision read from full-resolution cameras: exactly the scene state our
kinematic features lack. -> artifacts/cotfeat_<split>.json {clip: {cot_*: float}}"""
import glob, json, os, re, sys
import numpy as np
CUES = {
    "stop": r"\bstop(s|ped|ping)?\b(?! ?ped)", "remain_stopped": r"remain stopped|stay stopped|wait",
    "yield": r"\byield", "slow": r"\bslow", "accelerate": r"accelerat|speed up", "proceed": r"\bproceed|continue|go straight|drive through",
    "keep_distance": r"keep distance|following|maintain", "nudge_left": r"nudge (to the )?left", "nudge_right": r"nudge (to the )?right",
    "merge_left": r"merg\w* (to the )?left|lane change left|change lanes? (to the )?left", "merge_right": r"merg\w* (to the )?right|lane change right|change lanes? (to the )?right",
    "turn_left": r"turn(ing)? left|left turn", "turn_right": r"turn(ing)? right|right turn",
    "red": r"\bred\b", "green": r"\bgreen\b", "yellow": r"\byellow|amber", "light": r"light", "stop_sign": r"stop sign",
    "pedestrian": r"pedestrian|walking|person", "cyclist": r"cyclist|bicycl|bike", "construction": r"construction|cone|barrel|barrier",
    "stopped_vehicle": r"stopped (vehicle|car|truck)|parked", "lead_vehicle": r"lead vehicle", "intersection": r"intersection",
    "crosswalk": r"crosswalk", "blocking": r"blocking|block",
}
RX = {k: re.compile(v) for k, v in CUES.items()}


def feats(cots):
    cots = [c.lower() for c in cots if c]
    if not cots: return {}
    return {f"cot_{k}": float(np.mean([bool(r.search(c)) for c in cots])) for k, r in RX.items()}


if __name__ == "__main__":
    split = sys.argv[1]; root = sys.argv[2] if len(sys.argv) > 2 else f"a15_out/nonav/{split}"
    out = {}
    for f in glob.glob(f"{root}/*.npz"):
        z = np.load(f, allow_pickle=False)
        out[os.path.basename(f)[:-4]] = feats([str(x) for x in np.array(z["cot"]).ravel()])
    json.dump(out, open(f"artifacts/cotfeat_{split}.json", "w"))
    cov = np.mean([bool(v) for v in out.values()])
    print(split, len(out), "clips; with text", round(cov, 3), "; mean cue rates",
          {k: round(float(np.mean([v.get(f"cot_{k}", 0) for v in out.values()])), 3) for k in ("stop", "yield", "slow", "proceed", "red", "green", "pedestrian")})
