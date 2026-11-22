"""Stage-1 (VLM) index for Alpamayo on nuReasoning: the stage-2 samples (artifacts/a15ft_index_<split>.json,
all images present) whose reasoning frame carries a Driving annotation, with its one-sentence
"Reasoning trace" as the chain-of-causation target. -> artifacts/a15s1_index_<split>.json
rows: [clip, r, cot]"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor

split = sys.argv[1] if len(sys.argv) > 1 else "train"


def one(item):
    clip, rs = item
    fr = json.load(open(f"cache/{split}/{clip}/metadata.json"))["frames"]
    out = []
    for r in rs:
        p = fr[r].get("reasoning")
        if not p or not os.path.isfile(f"cache/{split}/{clip}/{p}"): continue
        dr = json.load(open(f"cache/{split}/{clip}/{p}")).get("Driving")
        if isinstance(dr, str) and dr.strip():
            try: dr = json.loads(dr)
            except json.JSONDecodeError: continue
        t = (dr or {}).get("Reasoning trace") if isinstance(dr, dict) else None
        if t and t.strip(): out.append([clip, r, " ".join(t.split())])
    return out


if __name__ == "__main__":
    rows = json.load(open(f"artifacts/a15ft_index_{split}.json"))
    by = {}
    for c, r in rows: by.setdefault(c, []).append(r)
    with ProcessPoolExecutor(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as ex:
        out = [x for o in ex.map(one, sorted(by.items()), chunksize=16) for x in o]
    json.dump(out, open(f"artifacts/a15s1_index_{split}.json", "w"))
    L = sorted(len(x[2].split()) for x in out)
    print(f"{split}: {len(out)} of {len(rows)} samples have a reasoning trace, {len({x[0] for x in out})} clips; "
          f"words p50 {L[len(L)//2]} p90 {L[int(.9*len(L))]}")
