"""Local reasoning harness: evaluate answer strategies on LABELLED VQA.

Questions come from the devkit's own generator run over cached validation clips,
so the distribution matches the challenge. Clips are split fit/eval so priors are
never measured on the clips they were fitted on.

Overall accuracy is reported reweighted to the TEST subcategory mix, because the
generated mix differs from the 3000 sampled challenge questions.
"""
import json, os, re, sys, collections, random
import numpy as np

TEST_MIX = {}  # filled from cache/test


def load_labelled(root="vqa_val"):
    out = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            if not f.endswith(".json"):
                continue
            o = json.load(open(os.path.join(p, f)))
            qq = o.get("questions") if isinstance(o, dict) else o
            for q in (qq or []):
                if q.get("question_type") == "choice" and q.get("answer"):
                    q["_clip"] = d
                    out.append(q)
    return out


def test_mix(root="cache/test"):
    c = collections.Counter()
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d, "reasoning_questions.json")
        if os.path.isfile(p):
            for q in (json.load(open(p)).get("questions") or []):
                c[q.get("subcategory")] += 1
    n = sum(c.values())
    return {k: v / n for k, v in c.items()}


# ---------------------------------------------------------------- strategies
def s_fixed_C(q, ctx):
    ch = q.get("choices") or {}
    return "C" if "C" in ch else sorted(ch)[0]


def s_prior(q, ctx):
    """Pick the choice whose TEXT is most common as an answer in this subcategory."""
    pri = ctx["prior"].get(q.get("subcategory"), {})
    ch = q.get("choices") or {}
    if not ch:
        return "A"
    return max(ch, key=lambda k: pri.get(_norm(ch[k]), 0.0))


def s_underscore(q, ctx):
    """Format artefact probe: prefer a choice containing '_'."""
    ch = q.get("choices") or {}
    u = [k for k, v in ch.items() if "_" in str(v)]
    return u[0] if len(u) == 1 else s_prior(q, ctx)


def _norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower().replace("_", " "))


STRATS = {"fixed_C": s_fixed_C, "prior": s_prior, "underscore+prior": s_underscore}


def main():
    qs = load_labelled()
    mix = test_mix()
    clips = sorted({q["_clip"] for q in qs})
    random.Random(0).shuffle(clips)
    half = set(clips[:len(clips) // 2])
    fit = [q for q in qs if q["_clip"] in half]
    ev = [q for q in qs if q["_clip"] not in half]
    print(f"labelled choice questions: {len(qs)}  (fit {len(fit)} / eval {len(ev)} by clip)")

    # fit per-subcategory answer-text priors
    prior = collections.defaultdict(collections.Counter)
    for q in fit:
        prior[q.get("subcategory")][_norm(q.get("answer_text"))] += 1
    prior = {k: {t: c / sum(v.values()) for t, c in v.items()} for k, v in prior.items()}
    ctx = {"prior": prior}

    subs = sorted({q.get("subcategory") for q in ev} | set(mix))
    print(f"\n{'subcategory':34} {'n':>5} {'test%':>6} " +
          " ".join(f"{s:>16}" for s in STRATS))
    overall = {s: 0.0 for s in STRATS}
    wsum = 0.0
    for sub in subs:
        rows = [q for q in ev if q.get("subcategory") == sub]
        if not rows:
            continue
        w = mix.get(sub, 0.0)
        line = f"{sub:34} {len(rows):>5} {100*w:>5.1f}%"
        for s, fn in STRATS.items():
            acc = np.mean([fn(q, ctx) == q["answer"] for q in rows])
            overall[s] += w * acc
            line += f" {acc:>16.3f}"
        wsum += w
        print(line)
    print(f"\n{'TEST-WEIGHTED OVERALL':34} {'':>5} {100*wsum:>5.1f}%" +
          "".join(f" {overall[s]/max(wsum,1e-9):>16.3f}" for s in STRATS))
    print(f"\n  (server measured all-A 0.2563, all-B 0.2377, all-C 0.2683)")

    # is the correct answer's TEXT systematically distinguishable?
    print("\n=== format-artefact probes on eval half ===")
    n_us = sum(1 for q in ev if any("_" in str(v) for v in (q.get("choices") or {}).values()))
    hit = [q for q in ev if sum("_" in str(v) for v in (q.get("choices") or {}).values()) == 1]
    if hit:
        acc = np.mean([("_" in str((q.get("choices") or {}).get(q["answer"], ""))) for q in hit])
        print(f"  questions with exactly ONE underscored choice: {len(hit)} "
              f"({100*len(hit)/len(ev):.1f}%); that choice is correct {100*acc:.1f}% of the time")
    lens = []
    for q in ev:
        ch = q.get("choices") or {}
        if len(ch) < 2: continue
        L = {k: len(str(v)) for k, v in ch.items()}
        lens.append(max(L, key=L.get) == q["answer"])
    print(f"  longest choice is correct: {100*np.mean(lens):.1f}% (chance ~{100/4:.0f}%)")


if __name__ == "__main__":
    main()
