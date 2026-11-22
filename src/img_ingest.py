"""Stream clip tars from HF and keep ONLY the listed key-frame images.

Each tar is piped straight into GNU tar with an explicit member list, so no
archive ever lands on disk; peak storage is just the kept JPEGs. --occurrence
lets tar stop as soon as every listed member is found, and curl then exits on
SIGPIPE, so we often stream less than the whole archive.

Run as a Slurm array: shard k of N processes every N-th clip. Resumable: a clip
is skipped if its _done marker exists. Every clip writes a provenance line.
"""
import argparse, json, os, subprocess, sys, time, hashlib
from concurrent.futures import ThreadPoolExecutor

BASE = "https://huggingface.co/datasets/nureasoning/nuReasoning/resolve/main/"


def one(row, out_root, auth_cfg):
    clip = row["clip"]
    dst = os.path.join(out_root, clip)
    done = os.path.join(dst, "_done.json")
    if os.path.isfile(done):
        return {"clip": clip, "status": "skipped"}
    os.makedirs(dst, exist_ok=True)
    lst = os.path.join(dst, "_members.txt")
    with open(lst, "w") as fh:
        fh.write("\n".join(row["members"]) + "\n")
    t0 = time.time()
    # strip the leading "<clip>/" so images land at dst/cameras/...
    # The token must NEVER appear in argv: on a shared node `ps` shows every
    # user's process arguments. It lives in a mode-600 curl config instead.
    # --speed-limit/--speed-time: abort a stream that stays under 2 MB/s for 45 s
    # (the first launch stalled 136 streams at ~0 B/s for minutes).
    cmd = (f'curl -sSL --fail --speed-limit 2000000 --speed-time 45 -K "{auth_cfg}" "{BASE}{row["tar"]}" '
           f'| tar -x --occurrence=1 -C "{dst}" --strip-components=1 -T "{lst}"')
    for attempt in range(3):
        p = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True)
        got = [m for m in row["members"]
               if os.path.isfile(os.path.join(dst, m.split("/", 1)[1]))]
        if len(got) == len(row["members"]):
            break
        time.sleep(10 * (attempt + 1))
    # curl exits 23 (write error) when tar stops early after finding everything: expected
    ok = len(got) == len(row["members"])
    rec = {"clip": clip, "status": "ok" if ok else "failed",
           "found": len(got), "wanted": len(row["members"]),
           "sec": round(time.time() - t0, 1), "rc": p.returncode,
           "stderr": p.stderr.strip()[-300:] if not ok else "",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if ok:
        rec["sha256"] = {m: hashlib.sha256(open(os.path.join(dst, m.split("/", 1)[1]), "rb").read()).hexdigest()
                         for m in row["members"]}
        json.dump(rec, open(done, "w"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="src/image_members.json")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", default="img")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    token = open(os.path.join(hf, "token")).read().strip()
    auth_cfg = os.path.join(hf, "curl_auth.cfg")
    old = os.umask(0o077)
    with open(auth_cfg, "w") as fh:
        fh.write(f'header = "Authorization: Bearer {token}"\n')
    os.umask(old)
    os.chmod(auth_cfg, 0o600)
    del token
    rows = json.load(open(a.members))[a.split][a.shard::a.nshards]
    if a.limit: rows = rows[:a.limit]
    out = os.path.join(a.out, a.split)
    print(f"[img] split={a.split} shard {a.shard}/{a.nshards} clips={len(rows)} workers={a.workers}", flush=True)
    t0 = time.time(); ok = fail = skip = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(lambda row: one(row, out, auth_cfg), rows)):
            ok += r["status"] == "ok"; skip += r["status"] == "skipped"; fail += r["status"] == "failed"
            if r["status"] == "failed":
                print(f"  FAIL {r['clip']} found {r['found']}/{r['wanted']} rc={r['rc']} {r['stderr']}", flush=True)
            if (i + 1) % 25 == 0 or i + 1 == len(rows):
                print(f"  [{i+1}/{len(rows)}] ok={ok} skip={skip} fail={fail} {time.time()-t0:.0f}s", flush=True)
    print(f"[img] DONE ok={ok} skip={skip} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
