"""Can we fetch ONLY chosen images from a clip tar without downloading it?

A tar has no index, but every member is a 512-byte header followed by its data,
so walking headers with small range requests yields an exact member index.
Then fetch only the key-frame images. Measures requests, bytes and wall time.
"""
import json, os, sys, time, urllib.request
sys.path.insert(0, "src")
from tarrange import _valid_header, BLOCK

TOK = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
path = sys.argv[1]
BASE = "https://huggingface.co/datasets/nureasoning/nuReasoning/resolve/main/"

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k): return None
op = urllib.request.build_opener(NoRedirect)
try:
    op.open(urllib.request.Request(BASE + path, method="HEAD",
            headers={"Authorization": f"Bearer {TOK}"}))
    cdn = BASE + path
except urllib.error.HTTPError as e:
    cdn = e.headers["Location"]                        # signed CDN URL, reuse it

def rng(a, b):
    r = urllib.request.Request(cdn, headers={"Range": f"bytes={a}-{b}"})
    with urllib.request.urlopen(r, timeout=60) as f: return f.read()

t0 = time.time(); pos = 0; nreq = 0; index = {}; pending = None
while True:
    blk = rng(pos, pos + BLOCK - 1); nreq += 1
    h = _valid_header(blk)
    if h is None: break
    name, size, tf = h
    data = pos + BLOCK
    if tf == b"L":
        pending = rng(data, data + size - 1).split(b"\0")[0].decode(); nreq += 1
    else:
        index[pending or name] = (data, size); pending = None
    pos = data + ((size + BLOCK - 1) // BLOCK) * BLOCK
walk = time.time() - t0
print(f"header walk: {len(index)} members, {nreq} requests, {walk:.1f}s "
      f"({1000*walk/nreq:.0f} ms/request)")
keyimgs = [k for k in index if "/cameras/" in k]
print(f"camera members: {len(keyimgs)}")
