"""Members to fetch for future-motion tracking: frames 90..100 (10 Hz, 1 s) of the camera each
future_motion_label question names, for validation and test. -> src/fm_members.json (same format
as src/image_members.json: {split: [{clip, tar, size, members}]})."""
import json, os, re, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RE_CAM = re.compile(r"in the ([a-z ]+?) camera")
FIRST, LAST = int(os.environ.get("FM_FIRST", 90)), int(os.environ.get("FM_LAST", 100))
OUT = os.environ.get("FM_OUT", "src/fm_members.json")
SUB = os.environ.get("FM_SUB", "future_motion_label")
base = json.load(open("src/image_members.json"))
out = {}
for split in ("test", "validation"):
    tar = {r["clip"]: r for r in base[split]}
    if split == "test":
        qfiles = {c: json.load(open(f"cache/test/{c}/reasoning_questions.json")).get("questions") or []
                  for c in os.listdir("cache/test") if os.path.isfile(f"cache/test/{c}/reasoning_questions.json")}
    else:
        qfiles = {}
        for f in glob.glob("vqa_val/*/*_vqa.json"):
            qfiles.setdefault(f.split("/")[1], []).extend(json.load(open(f)).get("questions") or [])
    rows = []
    for c, qs in sorted(qfiles.items()):
        cams = {RE_CAM.search(q.get("question", "")).group(1).strip().replace(" ", "_")
                for q in qs if q.get("subcategory") == SUB and RE_CAM.search(q.get("question", ""))}
        if not cams or c not in tar: continue
        fr = json.load(open(f"cache/{split}/{c}/metadata.json"))["frames"]
        mem = sorted({f"{c}/{fr[k]['sensors']['cameras'][cam]}" for cam in cams for k in range(FIRST, LAST + 1)
                      if fr[k]["sensors"]["cameras"].get(cam)})
        rows.append({"clip": c, "tar": tar[c]["tar"], "size": tar[c]["size"], "members": mem})
    out[split] = rows
    print(split, len(rows), "clips,", sum(len(r["members"]) for r in rows), "images")
json.dump(out, open(OUT, "w"))
