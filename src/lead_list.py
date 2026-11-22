"""2026-11-13, for the planning thread: clips where a lead vehicle is close (front camera, 0 < x <= 10 m, |y| <= 1.5 m)
and closing (> 0.5 m/s, approaching).
validation: from ALL annotated objects at the key frame (cache/validation/<clip>/reasoning/*.json, exact;
  x = box centre in the ego frame, so the bumper gap is ~2-2.5 m less).
validation_tracker / test: only objects named in future_motion questions (stated x, y) with the SAM 2.1 tracker
closing estimate (src/fm_rule.py closing, box height). On test this is a PARTIAL list: a clip with no such question
is unknown, not 'no lead'. Covered clips are listed so absence can be read correctly.
-> artifacts/lead_close.json"""
import glob, json, math, os, sys
sys.path.insert(0, "src")
import camgeo
from fm_rule import closing
VEH = {"car", "truck", "bus", "trailer", "construction_vehicle", "vehicle", "motorcycle", "emergency_vehicle"}
lead = lambda x, y: 0 < x <= 20 and abs(y) <= 1.5          # filtered into the two lists below
out = {"validation": {}, "validation_tracker": {}, "test": {}, "covered": {"validation_tracker": [], "test": []}}
cats = set()
def walk(o):
    if isinstance(o, dict):
        if "future_movement_description" in o and "position_3d_ego" in o: yield o
        for v in o.values(): yield from walk(v)
    elif isinstance(o, list):
        for v in o: yield from walk(v)
nclip_val = 0
for c in sorted(os.listdir("cache/validation")):           # ALL annotated objects at the key frame (frame ~100)
    fs = glob.glob(f"cache/validation/{c}/reasoning/*.json")
    if not fs: continue
    nclip_val += 1; seen = {}
    for fn in fs:
        R = json.load(open(fn))
        if abs(R["frame_index"] - 100) > 1: continue
        for o in walk(R):
            cat = o.get("category", ""); cats.add(cat)
            if not cat.startswith("vehicle") or "front" not in (o.get("camera_observations") or {}): continue
            x, y = o["position_3d_ego"]["x"], o["position_3d_ego"]["y"]
            cl = (o.get("geometric_relations") or {}).get("closing_speed_mps")
            if cl is None or not lead(x, y): continue
            if o["track_token"] not in seen or x < seen[o["track_token"]]["x"]:
                seen[o["track_token"]] = dict(x=round(x, 2), y=round(y, 2), closing=round(cl, 2), cat=cat)
    if seen: out["validation"][c] = min(seen.values(), key=lambda v: v["x"])   # nearest in-lane lead <= 20 m
for split, pref in (("validation_tracker", "runs_fm/sam21_full"), ("test", "runs_fm/sam21_test")):
    recs = {}
    for f in sorted(glob.glob(f"{pref}_s*of4.json")): recs.update(json.load(open(f)))
    src = "validation" if split.startswith("validation") else "test"; rigs = {}; cov = set()
    for qid, r in recs.items():
        cov.add(r["clip"])
        if r["cam"] != "front" or r["x"] is None or not lead(r["x"], r["y"]): continue
        if r["clip"] not in rigs: rigs[r["clip"]] = camgeo.Rig(json.load(open(f"cache/{src}/{r['clip']}/metadata.json")))
        c, why = closing(r, rigs[r["clip"]], "h")
        if why == "ok":
            prev = out[split].get(r["clip"])
            if prev is None or r["x"] < prev["x"]: out[split][r["clip"]] = dict(x=r["x"], y=r["y"], closing=round(c, 2))
    out["covered"][split] = sorted(cov)
# two lists per split: near10 = nearest lead centre <= 10 m (any closing); closing20 = <= 20 m and closing > 0.5 m/s
# (the requested '<= 10 m AND closing' is EMPTY on validation annotations: leads that close are queues of stopped cars)
for k in ("validation", "validation_tracker", "test"):
    d = out[k]; out[k] = {"near10": {c: v for c, v in d.items() if v["x"] <= 10},
                          "closing20": {c: v for c, v in d.items() if v["closing"] > 0.5},
                          "near10_closing": {c: v for c, v in d.items() if v["x"] <= 10 and v["closing"] > 0.5}}
json.dump(out, open("artifacts/lead_close.json", "w"), indent=1)
print(f"validation annotations ({nclip_val} clips with key-frame annotations):", {k: len(v) for k, v in out["validation"].items()})
for s_ in ("validation_tracker", "test"):
    print(f"{s_} (future_motion objects only; {len(out['covered'][s_])} clips covered):", {k: len(v) for k, v in out[s_].items()})
cv = set(out["covered"]["validation_tracker"])
for k in ("near10", "closing20"):
    t = set(out["validation_tracker"][k]); g = set(out["validation"][k]) & cv
    print(f"tracker vs truth on covered validation clips, {k}: tracker {len(t)}, truth {len(g)}, both {len(t & g)}")
