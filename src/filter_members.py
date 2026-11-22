import json, os, numpy as np
m = json.load(open("src/image_members.json"))
lab = set(str(n) for n in np.load("artifacts/cands_train.npz")["names"])
before = len(m["train"])
m["train"] = [r for r in m["train"] if r["clip"] in lab]
json.dump(m, open("src/image_members.json.tmp", "w"))
os.replace("src/image_members.json.tmp", "src/image_members.json")
print("train members:", before, "->", len(m["train"]), "(labeled route clips only)")
