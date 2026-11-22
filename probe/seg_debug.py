import json, os, sys, time
sys.path.insert(0, "src")
import tarindex as T
from concurrent.futures import ThreadPoolExecutor
tok = open(os.path.join(os.environ["HF_HOME"], "token")).read().strip()
r = json.load(open("src/image_members.json"))["validation"][300]
url = T.cdn_url(r["tar"], tok); size = r["size"]; K = 16; wanted = set(r["members"])
rgs = [T.Ranger(url) for _ in range(K)]
bounds = [size * k // K for k in range(K)]
t0 = time.time()
with ThreadPoolExecutor(K) as ex:
    starts = list(ex.map(lambda k: 0 if k == 0 else T._resync(rgs[k], bounds[k], size), range(K)))
t1 = time.time()
print(f"resync: {t1-t0:.1f}s  none={sum(s is None for s in starts)}  req={[x.nreq for x in rgs]}", flush=True)
starts = [s if s is not None else size for s in starts] + [size]
spans = [(starts[k+1]-starts[k])/1e6 for k in range(K)]
print("segment spans MB:", [round(s) for s in spans], flush=True)
def timed(k):
    a = time.time(); n0 = rgs[k].nreq
    f = T._walk(rgs[k], starts[k], starts[k+1], size, wanted)
    return k, time.time()-a, rgs[k].nreq-n0, len(f)
with ThreadPoolExecutor(K) as ex:
    res = list(ex.map(timed, range(K)))
t2 = time.time()
print(f"walk: {t2-t1:.1f}s", flush=True)
for k, dt, n, f in res:
    print(f"  seg {k:2d}: {dt:5.1f}s  {n:4d} req  {n/max(dt,1e-9):5.1f} req/s  found {f}", flush=True)
