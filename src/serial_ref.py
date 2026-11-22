import json, os, sys, time
sys.path.insert(0, "src")
from tarindex import fetch_members
tok = open(os.path.join(os.environ["HF_HOME"], "token")).read().strip()
r = json.load(open("src/image_members.json"))["validation"][300]
t0 = time.time(); print(fetch_members(r["tar"], r["size"], r["members"], tok, f"img_probe_serial/{r['clip']}"), f"{time.time()-t0:.0f}s", flush=True)
