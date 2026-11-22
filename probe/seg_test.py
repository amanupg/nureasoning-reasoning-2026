import json, os, sys, time, hashlib
sys.path.insert(0, "src")
from tarindex import fetch_members_parallel
tok = open(os.path.join(os.environ["HF_HOME"], "token")).read().strip()
rows = json.load(open("src/image_members.json"))["validation"][300:303]
for K in (24,):
    for r in rows:
        d = f"img_probe_seg{K}/{r['clip']}"
        t0 = time.time()
        got, nreq, nb = fetch_members_parallel(r["tar"], r["size"], r["members"], tok, d, K=K)
        dt = time.time() - t0
        # byte-identical to the serial walker's output?
        same = all(hashlib.sha256(open(f"{d}/{m.split('/',1)[1]}","rb").read()).digest() ==
                   hashlib.sha256(open(f"img_probe_serial/{r['clip']}/{m.split('/',1)[1]}","rb").read()).digest()
                   for m in r["members"])
        print(f"K={K:2d} {r['clip'][:36]}  {got}/16  {nreq} req  {nb/1e6:.1f} MB  {dt:.1f}s  identical_to_serial={same}", flush=True)
