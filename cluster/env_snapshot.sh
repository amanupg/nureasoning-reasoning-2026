#!/bin/bash
# One-time: pack the Python env into a single tar on the fileset (read sequentially by cluster/stage_env.sh).
# Rebuild after any pip install into env/.  (2026-11-06)
cd /path/to/cluster/scratch/nureasoning
t0=$(date +%s); tar -cf env_snapshot.tar.tmp env && mv env_snapshot.tar.tmp env_snapshot.tar
echo "[snapshot] $(du -sh env_snapshot.tar | cut -f1) in $(( $(date +%s) - t0 )) s"
