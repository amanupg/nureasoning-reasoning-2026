"""Read-only: files duplicated between two folders (>= 1 MB, same size, then same sha256).
usage: python cluster/dedupe_scan.py <dirA> <dirB> <out.tsv>"""
import hashlib, os, sys
from collections import defaultdict
A, B, out = sys.argv[1:4]
def walk(root):
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            try:
                if os.path.islink(p): continue
                s = os.stat(p).st_size
                if s >= 1 << 20: yield p, s
            except OSError: pass
sa = defaultdict(list); sb = defaultdict(list)
for p, s in walk(A): sa[s].append(p)
for p, s in walk(B): sb[s].append(p)
both = set(sa) & set(sb); print("size matches:", len(both), flush=True)
def h(p):
    x = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): x.update(b)
    return x.hexdigest()
n = tot = 0
with open(out, "w") as fo:
    fo.write("size_bytes\tsha256\tpath_A\tpath_B\n")
    for s in sorted(both, reverse=True):
        ha = defaultdict(list); hb = defaultdict(list)
        for p in sa[s]: ha[h(p)].append(p)
        for p in sb[s]: hb[h(p)].append(p)
        for k in set(ha) & set(hb):
            fo.write(f"{s}\t{k}\t{'|'.join(ha[k])}\t{'|'.join(hb[k])}\n"); n += 1; tot += s * len(hb[k])
print(f"duplicate groups {n}, bytes in B duplicated from A {tot / 1e9:.1f} GB", flush=True)
