"""Replace future_motion_label answers in a submission with the tracker rule (fm_track.py + fm_rule.py)
wherever the tracker is usable (>= 2 frames not touching the border, mask-box height); others unchanged.
Trajectories and all other answers are copied byte-for-byte.
usage: python src/fm_apply.py <in.json> <track_prefix> <out.json> <method-name> [--split test]
"""
import argparse, glob, hashlib, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camgeo, fm_rule as F


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("prefix"); ap.add_argument("out"); ap.add_argument("name")
    ap.add_argument("--split", default="test"); a = ap.parse_args()
    recs = {}
    for f in sorted(glob.glob(f"{a.prefix}_s*of*.json")): recs.update(json.load(open(f)))
    S = json.load(open(a.inp)); rigs = {}; n = changed = used = 0
    for c in S["clips"]:
        qs = {q["question_id"]: q for q in json.load(open(f"cache/{a.split}/{c['clip']}/reasoning_questions.json"))["questions"]}
        for x in c["answers"]:
            q = qs[x["question_id"]]
            if q.get("subcategory") != "future_motion_label": continue
            n += 1; r = recs.get(x["question_id"])
            if r is None or r.get("missing"): continue
            if r["clip"] not in rigs: rigs[r["clip"]] = camgeo.Rig(json.load(open(f"cache/{a.split}/{r['clip']}/metadata.json")))
            est = F.closing(r, rigs[r["clip"]], "h", edge=3, min_frames=2)[0]
            if math.isnan(est): continue
            letter = {v: k for k, v in q["choices"].items()}.get(F.rule(est))
            if letter is None: continue
            used += 1; changed += letter != x["answer"]; x["answer"] = letter
    S["meta"] = dict(S.get("meta", {}), method=a.name, answer_strategy=S["meta"].get("answer_strategy", "") + " + future_motion tracker rule")
    json.dump(S, open(a.out, "w"))
    print(f"future_motion questions {n}, tracker used {used}, answers changed {changed} -> {a.out}")
    print(hashlib.sha256(open(a.out, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
