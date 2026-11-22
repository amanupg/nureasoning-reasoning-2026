"""2026-11-14 future_motion: can a soft blend of the tracker, the GBDT and VLM run 1 beat the deployed hard rule
(tracker rule where usable, else GBDT)? Validation only; every input is out-of-sample for validation:
GBDT held-out probabilities (artifacts/gbdt_heldout_preds_probs.jsonl), VLM run 1 (train only, runs_vlm/qd4b_s3000_val*),
SAM 2.1 tracker closing estimate (runs_fm/sam21_full_*, no fitting to labels).
Blend model: HGB on [tracker closing estimate, usable flag, stated x, y, range, camera id, GBDT probs, VLM probs] by LABEL
TEXT (the 3 real labels), fitted on half the drive logs and scored on the other half (20 random halves, both ways),
paired against the deployed rule on the same held-out questions. Also simple variants without fitting.
usage: python src/fm_blend_eval.py
"""
import glob, json, math, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camgeo, fm_rule as F
from sklearn.ensemble import HistGradientBoostingClassifier
SUB = "future_motion_label"; LAB = list(F.LAB); CAMS = ["front", "front_left", "front_right", "left", "right", "back_left", "back_right", "back"]


def load(split, track_prefix, vlm_glob, gbdt_path, gbdt_key="probs"):
    recs = {}
    for f in sorted(glob.glob(f"{track_prefix}_s*of4.json")): recs.update(json.load(open(f)))
    G = {r["question_id"]: r for r in map(json.loads, open(gbdt_path)) if r["subcategory"] == SUB}
    V = {}
    for f in glob.glob(vlm_glob):
        if any(s in f for s in ("gray", "swap", "val150", "_p1_", "_p2_", "_p3_", "pavg", "valfus")): continue
        for l in open(f): r = json.loads(l); V[r["question_id"]] = r
    rigs, rows = {}, {}
    for qid, g in G.items():
        if qid not in V: continue
        c = g["clip"]; cache = f"cache/{split}/{c}"
        Q = {q["question_id"]: q for q in json.load(open(f"{cache}/reasoning_questions.json"))["questions"]} if split == "test" else None
        r = recs.get(qid); est = float("nan")
        if r is not None and not r.get("missing"):
            if c not in rigs: rigs[c] = camgeo.Rig(json.load(open(f"{cache}/metadata.json")))
            est = F.closing(r, rigs[c], "h", edge=3, min_frames=2)[0]
        rows[qid] = dict(g=g, v=V[qid], r=r, est=est, q=Q[qid] if Q else None)
    return rows


def choices_of(row, split):
    if row["q"] is not None: return row["q"]["choices"]
    return None


def main():
    # validation choices (letter -> text) from the VQA files
    CH = {}
    for f in glob.glob("vqa_val/*/*_vqa.json") + glob.glob("vqa_val_cl/*/*_vqa.json"):
        for q in json.load(open(f)).get("questions") or []:
            if q.get("subcategory") == SUB: CH[q["question_id"]] = q
    rows = load("validation", "runs_fm/sam21_full", os.environ.get("VLM_GLOB", "runs_vlm/qd4b_s3000_val*_s*of4.jsonl"), "artifacts/gbdt_heldout_preds_probs.jsonl")
    ids = sorted(i for i in rows if i in CH)
    print(f"validation future_motion with GBDT held-out + VLM run 1: {len(ids)}; tracker usable "
          f"{sum(not math.isnan(rows[i]['est']) for i in ids)}", flush=True)

    def lab_probs(p, q):            # letter probs -> label-text probs over LAB (other options dropped)
        out = {L: 0.0 for L in LAB}
        for k, v in p.items():
            t = q["choices"].get(k)
            if t in out: out[t] += v
        z = sum(out.values()) or 1.0
        return [out[L] / z for L in LAB]

    def feat(i):
        R = rows[i]; q = CH[i]; r = R["r"] or {}
        x, y = float(r.get("x") or np.nan), float(r.get("y") or np.nan)
        cam = CAMS.index(r["cam"]) if r.get("cam") in CAMS else -1
        return [R["est"], float(not math.isnan(R["est"])), x, y, math.hypot(x, y), cam] + lab_probs(R["g"]["probs"], q) + lab_probs(R["v"]["probs"], q)

    gold = np.array([CH[i]["choices"][CH[i]["answer"]] for i in ids]); log = np.array([rows[i]["g"]["clip"][:19] for i in ids])
    def pick(i, scores):            # label-text scores -> the letter's text actually offered
        q = CH[i]; offered = [t for t in q["choices"].values() if t in LAB]
        return max(offered, key=lambda t: scores[LAB.index(t)])
    dep = np.array([(F.rule(rows[i]["est"]) if not math.isnan(rows[i]["est"]) and F.rule(rows[i]["est"]) in CH[i]["choices"].values()
                     else CH[i]["choices"].get(max(rows[i]["g"]["probs"], key=rows[i]["g"]["probs"].get))) == g for i, g in zip(ids, gold)])
    gb = np.array([CH[i]["choices"].get(max(rows[i]["g"]["probs"], key=rows[i]["g"]["probs"].get)) == g for i, g in zip(ids, gold)])
    vl = np.array([CH[i]["choices"].get(rows[i]["v"]["pred"]) == g for i, g in zip(ids, gold)])
    us = np.array([not math.isnan(rows[i]["est"]) for i in ids])
    print(f"deployed (tracker rule else GBDT) {dep.mean():.3f} | GBDT {gb.mean():.3f} | VLM run 1 {vl.mean():.3f} | "
          f"on tracker-usable: deployed {dep[us].mean():.3f} GBDT {gb[us].mean():.3f} VLM {vl[us].mean():.3f}", flush=True)
    # 2026-11-17: paired clip bootstrap of simple routes vs deployed (VLM alone; tracker if far from threshold else VLM)
    def boot(b, name):
        L = np.array(log); ul = sorted(set(L)); rb = np.random.default_rng(0); idx = {l: np.where(L == l)[0] for l in ul}
        bs = [np.mean(np.concatenate([b[idx[l]] for l in rb.choice(ul, len(ul))]) - np.concatenate([dep[idx[l]] for l in rb.choice(ul, 0)] or [[]]).mean()) if False else
              (lambda s: b[s].mean() - dep[s].mean())(np.concatenate([idx[l] for l in rb.choice(ul, len(ul))])) for _ in range(2000)]
        print(f"  {name}: {b.mean():.3f} vs deployed {dep.mean():.3f}, diff {b.mean() - dep.mean():+.3f} "
              f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] wins {int((b & ~dep).sum())} losses {int((~b & dep).sum())}", flush=True)
    boot(vl, "VLM alone")
    for m in (0.5, 1.0):
        hy = np.array([(F.rule(rows[i]["est"]) if not math.isnan(rows[i]["est"]) and abs(abs(rows[i]["est"]) - 0.5) > m
                        and F.rule(rows[i]["est"]) in CH[i]["choices"].values() else CH[i]["choices"].get(rows[i]["v"]["pred"])) == g
                       for i, g in zip(ids, gold)])
        boot(hy, f"tracker if ||est|-0.5| > {m} else VLM")
    # variant: tracker rule only when it is far from the threshold, else GBDT+VLM average
    for m in (0.25, 0.5, 1.0):
        acc = []
        for i, g in zip(ids, gold):
            e = rows[i]["est"]; q = CH[i]
            if not math.isnan(e) and abs(abs(e) - 0.5) > m and F.rule(e) in q["choices"].values(): t = F.rule(e)
            else:
                s = np.add(lab_probs(rows[i]["g"]["probs"], q), lab_probs(rows[i]["v"]["probs"], q)); t = pick(i, s)
            acc.append(t == g)
        print(f"  tracker only if ||est|-0.5| > {m}, else mean(GBDT, VLM): {np.mean(acc):.3f}", flush=True)
    X = np.array([feat(i) for i in ids], float)
    logs = sorted(set(log)); rng = random.Random(0); d = []
    for rep in range(10):
        rng.shuffle(logs); h = set(logs[: len(logs) // 2]); mA = np.array([l in h for l in log])
        for fit, ev in ((mA, ~mA), (~mA, mA)):
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                               random_state=rep).fit(X[fit], gold[fit])
            P = m.predict_proba(X[ev]); cls = list(m.classes_)
            pr = np.array([pick(i, [P[k][cls.index(L)] if L in cls else 0 for L in LAB]) == g
                           for k, (i, g) in enumerate(zip(np.array(ids)[ev], gold[ev]))])
            d.append(pr.mean() - dep[ev].mean())
    print(f"HGB blend vs deployed, cross-fit by log (20 halves): {np.mean(d):+.4f} sd {np.std(d):.4f}, >0 in {np.mean(np.array(d) > 0):.2f}"
          f"  (x {len(ids)} / 2 questions per half)", flush=True)


if __name__ == "__main__":
    main()
