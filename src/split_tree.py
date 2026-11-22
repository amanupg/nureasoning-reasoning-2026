"""Split a nuVLA clip tree into two halves by DRIVE LOG (seed 0), for out-of-fold
models: model h0 trains on half 0 and predicts half 1, and vice versa.
Real per-clip dirs holding symlinks (os.walk does not follow symlinked dirs).
Writes <root>_h0/train, <root>_h1/train and artifacts/vla_halves.json."""
import json, os, random, sys
root = sys.argv[1] if len(sys.argv) > 1 else "vla_full"
clips = sorted(os.listdir(f"{root}/train"))
log_of = lambda c: "_".join(c.split("_")[:2])
logs = sorted({log_of(c) for c in clips}); random.Random(0).shuffle(logs)
half = {l: i % 2 for i, l in enumerate(logs)}
out = {"h0": [], "h1": []}
for c in clips:
    h = f"h{half[log_of(c)]}"; out[h].append(c)
    dst = f"{root}_{h}/train/{c}"; os.makedirs(dst, exist_ok=True)
    for name in os.listdir(f"{root}/train/{c}"):
        l = f"{dst}/{name}"
        if not os.path.lexists(l): os.symlink(os.path.realpath(f"{root}/train/{c}/{name}"), l)
json.dump({"split": "train", "by": "drive log, seed 0", "h0": out["h0"], "h1": out["h1"]},
          open("artifacts/vla_halves.json", "w"))
print({k: len(v) for k, v in out.items()}, "clips;", len(logs), "logs")
