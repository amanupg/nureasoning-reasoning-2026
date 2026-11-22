"""Archive local folders to the private HF repo amanupg/nr-archive, then verify every file
(path, size, and sha256 for LFS files) against the local copy. Prints VERIFIED only if all match.
usage: python cluster/archive_upload.py <local dir>=<repo path> [...]"""
import hashlib, os, sys
from huggingface_hub import HfApi
REPO = "amanupg/nr-archive"; api = HfApi()
pairs = [a.split("=", 1) for a in sys.argv[1:]]
for local, dest in pairs:
    print("uploading", local, "->", dest, flush=True)
    api.upload_folder(repo_id=REPO, folder_path=local, path_in_repo=dest, commit_message=f"archive {dest}",
                      ignore_patterns=["README.md"])      # adapter cards name a local base model -> rejected by the Hub
info = {f.rfilename: f for f in api.model_info(REPO, files_metadata=True).siblings}
bad = 0; n = 0; tot = 0
for local, dest in pairs:
    for d, _, fs in os.walk(local):
        for f in fs:
            if f == "README.md": continue
            p = os.path.join(d, f); r = os.path.join(dest, os.path.relpath(p, local)); n += 1; s = os.path.getsize(p); tot += s
            m = info.get(r)
            if m is None or m.size != s: print("MISMATCH size/missing", r, s, getattr(m, "size", None)); bad += 1; continue
            if m.lfs:
                h = hashlib.sha256()
                with open(p, "rb") as fh:
                    for b in iter(lambda: fh.read(16 << 20), b""): h.update(b)
                if h.hexdigest() != m.lfs.sha256: print("MISMATCH sha256", r); bad += 1
print(f"files {n}, {tot / 1e9:.1f} GB, mismatches {bad}")
print("VERIFIED" if bad == 0 else "NOT VERIFIED")
