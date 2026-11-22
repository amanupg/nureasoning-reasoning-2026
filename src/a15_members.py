"""Member list for the Alpamayo 1.5 pilot: 4 frames at 10 Hz (t0-0.3 s .. t0) for the
three forward cameras. Frames 100 already exist in img/; we fetch 97, 98, 99 (and 100
again so each clip dir is self-contained). Clips: a fixed random sample of validation."""
import argparse, json, os, random

CAMS = ["front_left", "front", "front_right"]

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="cache")
ap.add_argument("--base", default="artifacts/image_members.json")
ap.add_argument("--split", default="validation")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--frames", default="97,98,99,100")
ap.add_argument("--out", default="artifacts/a15_members.json")
a = ap.parse_args()
base = {r["clip"]: r for r in json.load(open(a.base))[a.split]}
clips = sorted(base)
random.Random(0).shuffle(clips)
rows = []
for c in clips[:a.n] if a.n > 0 else clips:
    fr = json.load(open(os.path.join(a.cache, a.split, c, "metadata.json")))["frames"]
    mem = [f"{c}/{fr[i]['sensors']['cameras'][cam]}" for i in map(int, a.frames.split(",")) for cam in CAMS]
    rows.append({"clip": c, "tar": base[c]["tar"], "size": base[c]["size"], "members": mem})
json.dump({a.split: rows}, open(a.out, "w"))
print(a.split, len(rows), "clips", len(rows[0]["members"]), "members each")
