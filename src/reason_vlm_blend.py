"""Per-subcategory blend of GBDT and VLM answer probabilities on the GBDT's held-out validation half:
p = (1 - a) * p_gbdt + a * p_vlm, a in {0, .25, .5, .75, 1} chosen per type on one half of the drive
logs and scored on the other (20 splits x 2). GBDT regressor types (no probabilities) use a one-hot
of the GBDT letter. image_region_from_3d_object is excluded (shipped answers there are calibration
geometry). Reports the test-mix-weighted held-out gain over GBDT, next to the switch-only rule."""
import json, glob, collections, random, os
import numpy as np
A_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
EXCLUDE = {"image_region_from_3d_object"}


def load():
    V = {}
    for f in glob.glob("reasoning_ws/eval/val_weak9_s*of8.jsonl"):
        for l in open(f):
            r = json.loads(l); V[(r["clip_dir"], r["question_id"])] = r
    G = [json.loads(l) for l in open("artifacts/gbdt_heldout_preds_probs.jsonl")]
    mix = collections.Counter()
    for d in sorted(os.listdir("cache/test")):
        p = os.path.join("cache/test", d, "reasoning_questions.json")
        if os.path.isfile(p):
            for q in (json.load(open(p)).get("questions") or []): mix[q.get("subcategory")] += 1
    tot = sum(mix.values())
    rows = [(g, V[(g["clip"], g["question_id"])]) for g in G if (g["clip"], g["question_id"]) in V and g["subcategory"] not in EXCLUDE]
    return rows, {k: v / tot for k, v in mix.items()}


def correct(g, v, a):
    L = sorted(v["probs"])
    pg = g["probs"] or {k: float(k == g["model"]) for k in L}
    s = {k: (1 - a) * pg.get(k, 0.0) + a * v["probs"][k] for k in L}
    return max(L, key=lambda k: s[k]) == g["gold"]


def main():
    rows, mix = load()
    lg = lambda c: "_".join(c.split("_")[:2])
    by = collections.defaultdict(list)
    for g, v in rows: by[g["subcategory"]].append((g, v))
    print(f"{'subcategory':32} {'n':>5} " + " ".join(f"a={a:<4}" for a in A_GRID))
    for s in sorted(by):
        print(f"{s:32} {len(by[s]):5d} " + " ".join(f"{np.mean([correct(g, v, a) for g, v in by[s]]):6.3f}" for a in A_GRID))
    logs = sorted({lg(g["clip"]) for g, v in rows}); gains, picks = [], collections.defaultdict(list)
    for seed in range(20):
        ll = logs[:]; random.Random(seed).shuffle(ll); P1 = set(ll[: len(ll) // 2])
        for first in (True, False):
            tot = 0.0
            for s, L in by.items():
                tr = [(g, v) for g, v in L if (lg(g["clip"]) in P1) == first]; te = [(g, v) for g, v in L if (lg(g["clip"]) in P1) != first]
                a = max(A_GRID, key=lambda a: np.mean([correct(g, v, a) for g, v in tr])); picks[s].append(a)
                tot += mix.get(s, 0) * (np.mean([correct(g, v, a) for g, v in te]) - np.mean([correct(g, v, 0.0) for g, v in te]))
            gains.append(tot)
    gains = np.array(gains)
    print(f"\nheld-out test-mix-weighted gain over GBDT (blend, a picked on the other half): {gains.mean():+.4f} (sd {gains.std():.4f}, >0 in {np.mean(gains > 0):.0%})")
    print("picked a per type:", {s: collections.Counter(v).most_common(2) for s, v in sorted(picks.items())})


if __name__ == "__main__":
    main()
