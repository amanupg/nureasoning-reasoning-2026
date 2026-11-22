"""How train/validation/test logs differ (date, vehicle, location) - metadata only."""
import json, os, collections, sys
R = "cache"
for s in ("train", "validation", "test"):
    C = collections.Counter(); D = collections.Counter(); L = collections.Counter(); logs = set()
    for c in os.listdir(f"{R}/{s}"):
        p = f"{R}/{s}/{c}/metadata.json"
        if not os.path.isfile(p): continue
        m = json.load(open(p)); ln = m.get("log_name", ""); logs.add(ln)
        D[ln[:7]] += 1; C[ln.split("_")[1] if "_" in ln else "?"] += 1; L[m.get("location", m.get("map_name", "?"))] += 1
    print(s, len(logs), "logs | months", sorted(D.items())[:12], "| loc", L.most_common(6), "| veh", len(C))
    if s == "train": print("  example", sorted(logs)[:2])
