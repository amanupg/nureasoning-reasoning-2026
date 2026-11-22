"""Fit per-subcategory answer-text priors from the labelled validation VQA.

Stored as normalized-text -> probability per subcategory. At answer time we pick
the choice whose text has the highest prior. Text is normalized (case, spacing,
underscores) because the generator emits both 'traffic_cone' and 'traffic cone'.
"""
import json, os, re, collections, sys

def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower().replace("_", " "))

root = sys.argv[1] if len(sys.argv) > 1 else "vqa_val"
out = sys.argv[2] if len(sys.argv) > 2 else "artifacts/answer_priors.json"
cnt = collections.defaultdict(collections.Counter)
n = 0
for d in sorted(os.listdir(root)):
    p = os.path.join(root, d)
    if not os.path.isdir(p): continue
    for f in sorted(os.listdir(p)):
        if not f.endswith(".json"): continue
        o = json.load(open(os.path.join(p, f)))
        qq = o.get("questions") if isinstance(o, dict) else o
        for q in (qq or []):
            if q.get("question_type") == "choice" and q.get("answer"):
                cnt[q.get("subcategory")][norm(q.get("answer_text"))] += 1
                n += 1
pri = {k: {t: c / sum(v.values()) for t, c in v.most_common()} for k, v in cnt.items()}
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"_source": root, "_n_questions": n, "priors": pri}, open(out, "w"), indent=1)
print(f"fitted {n} questions over {len(pri)} subcategories -> {out}")
for k in sorted(pri):
    top = list(pri[k].items())[:2]
    print(f"  {k:34} top: " + ", ".join(f"{t!r}={p:.3f}" for t, p in top))
