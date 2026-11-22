"""Submission with VLM answers for chosen subcategories (src/vlm_infer.py test output), everything else
(trajectories and other answers) copied from a base submission.
usage: python src/vlm_compose.py <base.json> <vlm prefix> <out.json> <method> [--subs all|a,b] [--keep future_motion_label]
"""
import argparse, glob, hashlib, json
from collections import Counter


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("base"); ap.add_argument("prefix"); ap.add_argument("out"); ap.add_argument("name")
    ap.add_argument("--subs", default="all"); ap.add_argument("--keep", default="")
    a = ap.parse_args()
    V = {}
    for f in glob.glob(f"{a.prefix}_s*of*.jsonl"):
        for l in open(f):
            r = json.loads(l); V[r["question_id"]] = r
    keep = set(x for x in a.keep.split(",") if x); subs = None if a.subs == "all" else set(a.subs.split(","))
    S = json.load(open(a.base)); ch = Counter(); used = Counter(); miss = 0
    for c in S["clips"]:
        for x in c["answers"]:
            r = V.get(x["question_id"])
            if r is None: miss += 1; continue
            s = r["subcategory"]
            if s in keep or (subs is not None and s not in subs): continue
            used[s] += 1; ch[s] += r["pred"] != x["answer"]; x["answer"] = r["pred"]
    S["meta"] = dict(S.get("meta", {}), method=a.name, answer_strategy=f"VLM {a.prefix} for {a.subs} (keep {a.keep or 'none'}); rest from {a.base.split('/')[-1]}")
    json.dump(S, open(a.out, "w"))
    print(f"VLM answers used {sum(used.values())}, changed {sum(ch.values())}, missing {miss}")
    for s in sorted(used): print(f"  {s:32s} used {used[s]:4d} changed {ch[s]:4d}")
    print(a.out, hashlib.sha256(open(a.out, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
