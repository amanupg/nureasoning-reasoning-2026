"""Fetch only the non-image tail of a nuReasoning clip tar via HTTP range requests.

Measured layout (verified on one clip per split, 2026-09-24):
  test  : reasoning_questions.json in the first 4 KB; metadata.json + ego_state/
          in a contiguous block in the last ~0.7 MB  (326x smaller than the tar)
  train : metadata.json + map.pkl + ego_state/ + annotations/ + reasoning/
  val   : same as train, contiguous in the last ~7-13 MB  (~61-66x smaller)

A tar has no central index, so we take a generous tail window, find 512-byte
aligned ustar headers inside it, and escalate the window if metadata.json is
not fully contained. Nothing is assumed about member order beyond "non-image
members are at the tail", and that assumption is CHECKED, not trusted: if
metadata.json is missing after escalation the clip is reported as a failure
rather than silently written as a partial cache entry.
"""
from __future__ import annotations

import io
import os
import struct
import tarfile
import urllib.request
from typing import Dict, Iterator, List, Optional, Tuple

BLOCK = 512
USTAR_MAGIC = b"ustar"


def _valid_header(block: bytes) -> Optional[Tuple[str, int, bytes]]:
    """Return (name, size, typeflag) if `block` is a plausible tar header."""
    if len(block) < BLOCK or block[257:262] != USTAR_MAGIC:
        return None
    # verify the header checksum; this is what rules out JPEG bytes that happen
    # to contain "ustar"
    stored = block[148:156]
    try:
        want = int(stored.split(b"\0")[0].split()[0], 8)
    except (ValueError, IndexError):
        return None
    probe = bytearray(block)
    probe[148:156] = b" " * 8
    if sum(probe) != want:
        return None
    name = block[0:100].split(b"\0")[0].decode("utf-8", "replace")
    prefix = block[345:500].split(b"\0")[0].decode("utf-8", "replace")
    if prefix:
        name = prefix + "/" + name
    try:
        size = int(block[124:136].split(b"\0")[0].strip() or b"0", 8)
    except ValueError:
        return None
    return name, size, block[156:157]


def walk_partial_tar(buf: bytes) -> Iterator[Tuple[str, int, int, bool]]:
    """Yield (name, data_offset_in_buf, size, is_file) for headers found in `buf`.

    `buf` must begin on a 512-byte boundary relative to the start of the tar.

    These archives are GNU tar with paths over 100 characters, so real names
    arrive in a preceding ``././@LongLink`` (typeflag 'L') member and the
    following header carries a TRUNCATED name. Missing this silently drops
    every ego_state/annotations/reasoning file, whose paths are 101+ chars.
    """
    pos = 0
    n = len(buf)
    pending_name: Optional[str] = None
    while pos + BLOCK <= n:
        hdr = _valid_header(buf[pos:pos + BLOCK])
        if hdr is None:
            pos += BLOCK          # we started inside file data; keep scanning
            pending_name = None
            continue
        name, size, typeflag = hdr
        data = pos + BLOCK
        nxt = data + ((size + BLOCK - 1) // BLOCK) * BLOCK
        if typeflag == b"L":                      # GNU long name for the NEXT member
            if data + size <= n:
                pending_name = buf[data:data + size].split(b"\0")[0].decode("utf-8", "replace")
            else:
                pending_name = None
            pos = nxt
            continue
        if pending_name:
            name = pending_name
            pending_name = None
        yield name, data, size, typeflag in (b"0", b"\0")
        pos = nxt


def http_range(url: str, token: str, start: Optional[int], end: Optional[int],
               suffix_len: Optional[int] = None, timeout: int = 120) -> bytes:
    if suffix_len is not None:
        rng = f"bytes=-{suffix_len}"
    else:
        rng = f"bytes={start}-{end}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Range": rng,
        "User-Agent": "nureasoning-ingest/1.0",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if r.status not in (200, 206):
            raise IOError(f"unexpected status {r.status} for {rng}")
        return r.read()
