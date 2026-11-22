"""Per-type probability fusion p = (1 - a) * p_gbdt + a * p_vlm on validation (GBDT held-out log x VLM
held-out predictions). a is chosen on one half of the drive logs and scored on the other (20 random
log splits x both directions); also reports the a chosen on all of validation.
usage: python src/vlm_fuse.py <vlm val prefix> [--write alphas.json]
"""
import argparse, glob, json, random
import numpy as np

GRID = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("prefix"); ap.add_argument("--write", default=""); a = ap.parse_args()
    V = {}
    for f in glob.glob(f"{a.prefix}_s*of*.jsonl"):
        for l in open(f): r = json.loads(l); V[r["question_id"]] = r
    G = {}
    for l in open("artifacts/gbdt_heldout_preds_probs.jsonl"):
        d = json.loads(l)
        if d.get("probs") and d["question_id"] in V: G[d["question_id"]] = d
    by = {}
    for q, g in G.items(): by.setdefault(g["subcategory"], []).append(q)
    log = lambda q: G[q]["clip"].split("_")[1]
    def acc(qs, al):
        ok = 0
        for q in qs:
            pg, pv = G[q]["probs"], V[q]["probs"]
            s = {k: (1 - al) * pg.get(k, 0) + al * pv.get(k, 0) for k in pv}
            ok += max(sorted(s), key=lambda k: s[k]) == G[q]["gold"]
        return ok / len(qs)
    out = {}
    print(f"{'type':32s}   n   GBDT   VLM   best-a(all)   cross-fitted fused  gain vs GBDT")
    for sub, qs in sorted(by.items()):
        logs = sorted({log(q) for q in qs}); held = []
        for seed in range(20):
            r = random.Random(seed); A = set(r.sample(logs, len(logs) // 2))
            for fit_half in (True, False):
                fit = [q for q in qs if (log(q) in A) == fit_half]; ev = [q for q in qs if (log(q) in A) != fit_half]
                if not fit or not ev: continue
                al = max(GRID, key=lambda x: (acc(fit, x), -abs(x - 0.5)))
                held.append((acc(ev, al) - acc(ev, 0.0), acc(ev, al)))
        best = max(GRID, key=lambda x: (acc(qs, x), -abs(x - 0.5)))
        h = np.array(held)
        print(f"{sub:32s} {len(qs):4d}  {acc(qs, 0):.3f} {acc(qs, 1):.3f}   {best:.1f} ({acc(qs, best):.3f})     "
              f"{h[:, 1].mean():.3f}            {h[:, 0].mean():+.3f} (positive in {np.mean(h[:, 0] > 0):.0%})")
        out[sub] = {"alpha": best, "crossfit_gain": float(h[:, 0].mean()), "pos_share": float(np.mean(h[:, 0] > 0))}
    if a.write: json.dump(out, open(a.write, "w"), indent=1); print("->", a.write)


if __name__ == "__main__":
    main()
