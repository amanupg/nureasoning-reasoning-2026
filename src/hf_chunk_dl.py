"""Parallel byte-range downloader for public Hugging Face repos (2026-11-13: anonymous downloads from Insomnia compute
nodes are throttled to ~0.4 MB/s PER CONNECTION, so throughput needs many connections per file). Each file is split
into chunks fetched with HTTP Range requests and written in place (os.pwrite) into a preallocated file; finished
chunks are recorded in <out>/.parts/ so a requeued job resumes. No token is used (public repos only).
usage: python src/hf_chunk_dl.py <repo id> <out dir> [--patterns 'a,b'] [--workers 96] [--chunk-mb 32]
"""
import argparse, fnmatch, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def get(url, rng=None, tries=30):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if rng and len(data) != rng[1] - rng[0] + 1: raise IOError(f"short read {len(data)}")
            return data
        except Exception as e:
            if t == tries - 1: raise
            time.sleep(min(30, 2 + 2 * t))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("repo"); ap.add_argument("out")
    ap.add_argument("--patterns", default="*.json,*.jinja,*.txt,transformer/*,vision_encoder/*,text_tokenizer/*")
    ap.add_argument("--workers", type=int, default=96); ap.add_argument("--chunk-mb", type=int, default=32)
    a = ap.parse_args()
    meta = json.loads(get(f"https://huggingface.co/api/models/{a.repo}?blobs=true"))
    pats = a.patterns.split(","); CH = a.chunk_mb << 20
    files = [(s["rfilename"], s.get("size") or 0) for s in meta["siblings"]
             if any(fnmatch.fnmatch(s["rfilename"], p) for p in pats)
             and s["rfilename"].count("/") <= 1]
    parts = os.path.join(a.out, ".parts"); os.makedirs(parts, exist_ok=True)
    jobs = []
    for f, size in files:
        dst = os.path.join(a.out, f); os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            with open(dst, "wb") as h: h.truncate(size)
        n = max(1, -(-size // CH))
        for i in range(n):
            mark = os.path.join(parts, f.replace("/", "__") + f".{i}")
            if not os.path.exists(mark): jobs.append((f, dst, size, i, mark))
    tot = sum(s for _, s in files)
    print(f"[dl] {a.repo}: {len(files)} files, {tot / 1e9:.1f} GB, {len(jobs)} chunks to fetch", flush=True)
    url = lambda f: f"https://huggingface.co/{a.repo}/resolve/main/{f}"
    t0 = time.time(); done = 0

    def one(j):
        f, dst, size, i, mark = j
        lo = i * CH; hi = min(size, lo + CH) - 1
        data = get(url(f), (lo, hi)) if size > 0 else get(url(f))
        fd = os.open(dst, os.O_WRONLY)
        try: os.pwrite(fd, data, lo)
        finally: os.close(fd)
        open(mark, "w").close(); return len(data)

    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for k, fu in enumerate(as_completed(futs)):
            done += fu.result()
            if (k + 1) % 50 == 0 or k + 1 == len(futs):
                el = time.time() - t0; print(f"  {k + 1}/{len(futs)} chunks, {done / 1e9:.2f} GB, {done / el / 1e6:.1f} MB/s", flush=True)
    for f, size in files:                                   # sanity: sizes match the API
        assert os.path.getsize(os.path.join(a.out, f)) == size, f
    print(f"[dl] DONE {a.repo} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
