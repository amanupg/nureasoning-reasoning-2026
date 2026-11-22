"""Build nuVLA's expected clip layout from our two trees, as symlinks only:
cache/<split>/<clip>/{metadata.json, ego_state, annotations, reasoning, map.pkl}
+ img/<split>/<clip>/cameras  ->  vla_data/<split>/<clip>/..."""
import os, sys
split = sys.argv[1]; n = 0
IMG = sys.argv[2] if len(sys.argv) > 2 else "img"; OUT = sys.argv[3] if len(sys.argv) > 3 else "vla_data"
for clip in sorted(os.listdir(f"cache/{split}")):
    src, img = f"cache/{split}/{clip}", f"{IMG}/{split}/{clip}/cameras"
    if not (os.path.isfile(f"{src}/metadata.json") and os.path.isfile(f"{IMG}/{split}/{clip}/_done.json")): continue
    dst = f"{OUT}/{split}/{clip}"; os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name.startswith("_") or name == "cameras": continue
        l = f"{dst}/{name}"
        if not os.path.lexists(l): os.symlink(os.path.abspath(f"{src}/{name}"), l)
    if not os.path.lexists(f"{dst}/cameras"): os.symlink(os.path.abspath(img), f"{dst}/cameras")
    n += 1
print(split, n, "clips")
