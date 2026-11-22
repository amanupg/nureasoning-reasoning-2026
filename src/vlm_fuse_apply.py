"""Apply per-type fusion p = (1 - a) * p_gbdt + a * p_vlm to TEST answers of a base submission.
usage: python src/vlm_fuse_apply.py <base.json> <vlm test prefix> <out.json> <method> --alphas sub:a,sub:a
GBDT test probabilities: artifacts/gbdt_test_probs.jsonl (src/gbdt_test_probs.py, shipped v7.1 bundle).
"""
import argparse, glob, hashlib, json
from collections import Counter

ap = argparse.ArgumentParser(); ap.add_argument("base"); ap.add_argument("prefix"); ap.add_argument("out"); ap.add_argument("name")
ap.add_argument("--alphas", required=True); ap.add_argument("--check", default="")
a = ap.parse_args()
AL = {k: float(v) for k, v in (x.split(":") for x in a.alphas.split(","))}
V = {}
for f in glob.glob(f"{a.prefix}_s*of*.jsonl"):
    for l in open(f): r = json.loads(l); V[r["question_id"]] = r
G = {json.loads(l)["question_id"]: json.loads(l) for l in open("artifacts/gbdt_test_probs.jsonl")}
if a.check:      # GBDT test argmax must reproduce the answers that were scored (for the fused types)
    C = {x["question_id"]: x["answer"] for c in json.load(open(a.check))["clips"] for x in c["answers"]}
    for s in AL:
        qs = [q for q, g in G.items() if g["subcategory"] == s]
        print(f"  check {s:32s} GBDT argmax == {a.check.split('/')[-1][:20]}: {sum(G[q]['pred'] == C[q] for q in qs)}/{len(qs)}")
S = json.load(open(a.base)); ch = Counter(); used = Counter()
for c in S["clips"]:
    for x in c["answers"]:
        g, v = G.get(x["question_id"]), V.get(x["question_id"])
        if not g or not v or g["subcategory"] not in AL: continue
        al = AL[g["subcategory"]]; s = {k: (1 - al) * g["probs"].get(k, 0) + al * v["probs"].get(k, 0) for k in v["probs"]}
        new = max(sorted(s), key=lambda k: s[k]); used[g["subcategory"]] += 1; ch[g["subcategory"]] += new != x["answer"]; x["answer"] = new
S["meta"] = dict(S.get("meta", {}), method=a.name, answer_strategy=S["meta"].get("answer_strategy", "") + f"; fused GBDT x VLM {a.alphas}")
json.dump(S, open(a.out, "w"))
for s in sorted(used): print(f"  {s:32s} fused {used[s]:4d} changed vs base {ch[s]:4d}")
print(a.out, hashlib.sha256(open(a.out, "rb").read()).hexdigest())
