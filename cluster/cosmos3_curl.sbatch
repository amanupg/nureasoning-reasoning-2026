#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -c 16
#SBATCH --mem=100G
#SBATCH -t 10:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/cosmos3_curl_%j.out
# 2026-11-13: huggingface_hub downloads of Cosmos3 stalled; anonymous HF is ~0.4 MB/s PER CONNECTION here, so this uses
# src/hf_chunk_dl.py (parallel HTTP Range chunks, resumable) instead of curl with the
# public (not gated) repos, so anonymous (the stored OAuth token in curl_auth.cfg expired 11-13), 8 files in
# parallel, resumable (-C -) into dl/Cosmos3-<r>/ on the fileset (survives requeue), then extracts the understanding
# tower (src/cosmos3_convert.py) to models/cosmos3_<r>_vlm. usage: sbatch cluster/cosmos3_curl.sbatch Nano [Super]
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
for r in "$@"; do
  low=$(echo $r | tr A-Z a-z); D=$P/dl/Cosmos3-$r; mkdir -p $D
  [ -f models/cosmos3_${low}_vlm/model.safetensors.index.json ] && { echo "[skip] $r converted"; continue; }
  t0=$(date +%s)
  env/bin/python -u src/hf_chunk_dl.py nvidia/Cosmos3-$r $D --workers ${WORKERS:-96} || exit 1
  bad=$(find $D -path $D/.parts -prune -o -type f -size -200c -print | wc -l); echo "[curl] tiny files: $bad"
  echo "[curl] $r done in $(( $(date +%s) - t0 )) s, $(du -sh $D | cut -f1)"
  env/bin/python -u src/cosmos3_convert.py $D models/cosmos3_${low}_vlm.tmp && mv models/cosmos3_${low}_vlm.tmp models/cosmos3_${low}_vlm && du -sh models/cosmos3_${low}_vlm || exit 1
done
echo PREP_DONE
