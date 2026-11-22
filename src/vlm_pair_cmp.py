"""2026-11-14: paired per-type validation comparison of two VLM runs on the same questions (raw argmax accuracy),
with a bootstrap CI per type by clip, and the test-share-weighted net (x test count per type).
usage: python src/vlm_pair_cmp.py <run A val glob> <run B val glob>"""
import json, glob, collections, sys, numpy as np
sys.path.insert(0, "src"); from vlm_data import TEST_SHARE
SKIP = ("gray", "swap", "val150", "_p1_", "_p2_", "_p3_", "pavg", "valfus")
def load(pat):
    D = {}
    for f in glob.glob(pat):
        if any(s in f for s in SKIP): continue
        for l in open(f): r = json.loads(l); D[r["question_id"]] = r
    return D
A, B = load(sys.argv[1]), load(sys.argv[2]); ids = sorted(i for i in B if i in A)
print(f"A {len(A)} B {len(B)} paired {len(ids)}")
T = collections.defaultdict(list)
for i in ids: T[B[i]["subcategory"]].append((B[i]["clip"], A[i]["pred"] == A[i]["gold"], B[i]["pred"] == B[i]["gold"]))
rng = np.random.default_rng(0); tot = 0.0
for s, rows in sorted(T.items(), key=lambda x: -TEST_SHARE.get(x[0], 0)):
    a = np.array([r[1] for r in rows], float); b = np.array([r[2] for r in rows], float); d = b - a
    clips = sorted({r[0] for r in rows}); byc = collections.defaultdict(list)
    for k, r in enumerate(rows): byc[r[0]].append(k)
    bs = [np.mean(np.concatenate([d[byc[c]] for c in rng.choice(clips, len(clips))])) for _ in range(2000)]
    nt = TEST_SHARE.get(s, 0) * 3000; tot += d.mean() * nt
    print(f"{s[:30]:30s} n {len(rows):4d} A {a.mean():.3f} B {b.mean():.3f} net {int(d.sum()):+4d} "
          f"CI [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] x test {nt:4.0f} = {d.mean() * nt:+.1f}")
print(f"test-weighted net (raw argmax, before fusion) {tot:+.1f} of 3000")
