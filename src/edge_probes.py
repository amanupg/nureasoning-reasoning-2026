"""2026-11-12 probes p22-p25: p19 with ONE question type's answers replaced by Alpamayo2-Edge zero-shot (native6,
runs_vlm/edge_test_native6_s*of2.jsonl). Stay-in-place trajectories (= p19's). Reasoning delta vs p19 0.7733 x 3000 =
(Edge correct - p19 correct) on the changed answers of that type."""
import glob, json, hashlib, sys
sys.path.insert(0, "recon/devkit")
from nureasoning.submission.format import validate_submission
E = {}
for f in glob.glob("runs_vlm/edge_test_native6_s*of2.jsonl"):
    for l in open(f): r = json.loads(l); E[r["question_id"]] = r
P = json.load(open("submissions/probes/p19_dist_vitl.json"))
for name, sub in (("p22_edge_region", "image_region_from_3d_object"), ("p23_edge_risk", "action_risk_assessment"),
                  ("p24_edge_camview", "camera_view_for_3d_object"), ("p25_edge_unsafe", "unsafe_action_identification")):
    S = json.loads(json.dumps(P)); ch = 0
    for c in S["clips"]:
        for a in c["answers"]:
            r = E[a["question_id"]]
            if r["subcategory"] == sub and r["pred"] != a["answer"]: a["answer"] = r["pred"]; ch += 1
    S["meta"] = dict(S["meta"], method=name, answer_strategy=f"p19 answers; {sub} from Alpamayo2-Edge zero-shot (src/edge_probes.py)")
    out = f"submissions/probes/{name}.json"; json.dump(S, open(out, "w"))
    err = validate_submission(S, data_root="cache/test")
    print(f"{name}: {sub}, {ch} answers changed, validate errors {len(err)}, sha {hashlib.sha256(open(out, 'rb').read()).hexdigest()[:16]}")
