"""Verify a range-fetched clip cache against the full tar, byte for byte.

Playbook rule: a derived cache is not trusted until it has been checked against
the raw source, and the check is not trusted until it has been shown to fail on
a deliberately corrupted input.
"""
import hashlib, os, sys, tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tarrange import walk_partial_tar, BLOCK


def members_from_full_tar(tar_path: str) -> dict:
    out = {}
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            rel = m.name.split("/", 1)[-1]
            if rel.startswith("cameras/") or rel.startswith("lidar"):
                continue
            out[rel] = hashlib.sha256(tf.extractfile(m).read()).hexdigest()
    return out


def members_from_tail(tar_path: str, tail_mb: float) -> dict:
    size = os.path.getsize(tar_path)
    start = max(0, (size - int(tail_mb * 1024 * 1024)) // BLOCK * BLOCK)
    with open(tar_path, "rb") as fh:
        fh.seek(start); buf = fh.read()
    head = open(tar_path, "rb").read(8192)
    out = {}
    for blob in (head, buf):
        for name, off, sz, is_file in walk_partial_tar(blob):
            rel = name.split("/", 1)[-1]
            if not is_file or rel.startswith("cameras/") or rel.startswith("lidar"):
                continue
            if off + sz <= len(blob):
                out[rel] = hashlib.sha256(blob[off:off + sz]).hexdigest()
    return out


def main():
    tar_path, tail_mb = sys.argv[1], float(sys.argv[2])
    truth = members_from_full_tar(tar_path)
    got = members_from_tail(tar_path, tail_mb)
    missing = sorted(set(truth) - set(got))
    extra = sorted(set(got) - set(truth))
    bad = sorted(k for k in set(truth) & set(got) if truth[k] != got[k])
    print(f"{os.path.basename(tar_path)[:52]}  tail={tail_mb} MB")
    print(f"  full-tar non-image members : {len(truth)}")
    print(f"  recovered from tail+head   : {len(got)}")
    print(f"  missing={len(missing)} extra={len(extra)} hash-mismatch={len(bad)}")
    if missing: print(f"    missing e.g. {missing[:4]}")
    if bad:     print(f"    mismatch e.g. {bad[:4]}")
    ok = not missing and not bad
    print(f"  VERDICT: {'PASS - byte identical' if ok else 'FAIL'}")

    # negative control: the check must be able to fail
    corrupted = dict(got)
    if corrupted:
        k = sorted(corrupted)[0]
        corrupted[k] = "0" * 64
        detects = any(truth[x] != corrupted[x] for x in set(truth) & set(corrupted))
        print(f"  negative control (corrupt {k}): "
              f"{'detected' if detects else 'NOT DETECTED - gate is broken'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
