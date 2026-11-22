"""ORACLE driving-decision features (annotation labels; train/validation only).
Measures what a perfect decision head would be worth to the chooser.
Output uses the percfeat layout (clip matrix) so scorer148 PERC=clip can load it
via PERC_FILE=decfeat."""
import collections, glob, json, os, sys
import numpy as np


def decision(cp):
    fr = json.load(open(os.path.join(cp, "metadata.json")))["frames"]
    for k in (100, 101, 99):
        r = fr[k].get("reasoning")
        if r and os.path.isfile(os.path.join(cp, r)):
            dr = json.load(open(os.path.join(cp, r))).get("Driving")
            d = dr.get("Driving decision") if isinstance(dr, dict) else None
            if isinstance(d, dict):
                return d.get("Longitudinal"), d.get("Lateral")
            return None, None
    return None, None


def main():
    D = {s: {os.path.basename(c): decision(c) for c in sorted(glob.glob(f"cache/{s}/*"))} for s in ("train", "validation")}
    lon = collections.Counter(v[0] for v in D["train"].values()); lat = collections.Counter(v[1] for v in D["train"].values())
    print("longitudinal:", lon.most_common()); print("lateral:", lat.most_common())
    L = [k for k, _ in lon.most_common() if k]; A = [k for k, _ in lat.most_common() if k]
    for s, dd in D.items():
        names = sorted(n for n, v in dd.items() if v[0] is not None)
        M = np.array([[float(dd[n][0] == k) for k in L] + [float(dd[n][1] == k) for k in A] for n in names], np.float32)
        np.savez_compressed(f"artifacts/decfeat_{s}.npz", names=np.array(names), clip=M,
                            clr=np.zeros((len(names), 148, 2), np.float32), clip_cols=np.array(L + A))
        print(s, len(names), "clips with a decision label")


if __name__ == "__main__":
    main()
