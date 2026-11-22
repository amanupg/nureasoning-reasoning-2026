#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 10:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/hf_get_%j.out
# 2026-11-14: download a public (not gated) HF model repo into models/<name> with src/hf_chunk_dl.py (parallel Range
# chunks; anonymous HF is ~0.4 MB/s per connection from compute nodes), resumable into dl/<name>, then moved.
# usage: sbatch cluster/hf_get.sbatch Qwen/Qwen3.8-27B qwen38_27b
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
R=$1; N=$2; D=$P/dl/$N
[ -f models/$N/config.json ] && { echo "[skip] models/$N exists"; echo GET_DONE; exit 0; }
t0=$(date +%s)
env/bin/python -u src/hf_chunk_dl.py $R $D --workers ${WORKERS:-96} --patterns "*.json,*.jinja,*.txt,*.safetensors,*.model" || exit 1
rm -rf $D/.parts && mv $D models/$N && du -sh models/$N && echo "[get] $R in $(( $(date +%s) - t0 )) s" && echo GET_DONE
