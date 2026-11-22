"""Structured web research through the Parallel Task API (key read from nureasoning/.env, never printed).
Each job: a question + a JSON output schema -> answer with per-field citations and confidence.
usage: python src/pq_research.py <jobs.json> <out_dir>      (jobs: [{id, processor, input, schema}])
Resumable: a job whose <out_dir>/<id>.json exists is skipped. Prints a cost estimate first."""
import json, os, sys, time, urllib.request

PRICE = {"lite": .005, "base": .01, "core": .025, "core2x": .05, "pro": .10, "ultra": .30, "ultra2x": .60}
KEY = [l.split("=", 1)[1].strip() for l in open(os.path.join(os.path.dirname(__file__), "..", ".env"))
       if l.startswith("PARALLEL_API_KEY=")][0]


def call(method, path, body=None):
    r = urllib.request.Request("https://api.parallel.ai" + path, method=method,
                               headers={"x-api-key": KEY, "Content-Type": "application/json"},
                               data=json.dumps(body).encode() if body is not None else None)
    return json.load(urllib.request.urlopen(r, timeout=600))


def main():
    jobs, out = json.load(open(sys.argv[1])), sys.argv[2]; os.makedirs(out, exist_ok=True)
    todo = [j for j in jobs if not os.path.isfile(os.path.join(out, j["id"] + ".json"))]
    print(f"{len(todo)} jobs, estimated ${sum(PRICE[j['processor']] for j in todo):.2f}", flush=True)
    runs = {}
    for j in todo:
        spec = {"output_schema": {"type": "json", "json_schema": j["schema"]}}
        r = call("POST", "/v1/tasks/runs", {"input": j["input"], "processor": j["processor"], "task_spec": spec})
        runs[j["id"]] = r["run_id"]; print("submitted", j["id"], j["processor"], flush=True)
    while runs:
        time.sleep(30)
        for jid, rid in list(runs.items()):
            st = call("GET", f"/v1/tasks/runs/{rid}")
            if st.get("status") in ("completed", "failed", "cancelled"):
                res = call("GET", f"/v1/tasks/runs/{rid}/result") if st["status"] == "completed" else st
                json.dump(res, open(os.path.join(out, jid + ".json"), "w"), indent=1)
                print("done", jid, st["status"], flush=True); del runs[jid]


if __name__ == "__main__":
    main()
