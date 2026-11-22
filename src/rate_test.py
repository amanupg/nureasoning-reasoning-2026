"""Is the ~10 req/s ceiling per signed URL or per account? Hit N different clip
URLs concurrently with 1536-byte range reads for 30 s and count completions."""
import json, os, sys, time, threading
sys.path.insert(0, "src")
from tarindex import cdn_url, Ranger
tok = open(os.path.join(os.environ["HF_HOME"], "token")).read().strip()
rows = json.load(open("src/image_members.json"))["validation"][200:264]
for N, per_url in ((1, 1), (1, 8), (8, 1), (32, 1)):
    urls = [cdn_url(r["tar"], tok) for r in rows[:N]]
    counts = [0] * (N * per_url); stop = time.time() + 30
    def worker(i):
        rg = Ranger(urls[i // per_url]); pos = (i % per_url) * 10_000_000
        while time.time() < stop:
            rg.get(pos, pos + 1535); counts[i] += 1; pos += 1536
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(N * per_url)]
    [t.start() for t in ts]; [t.join() for t in ts]
    tot = sum(counts)
    print(f"{N:2d} URL(s) x {per_url} conn each: {tot/30:6.1f} req/s total, {tot/30/(N*per_url):5.1f} per connection", flush=True)
