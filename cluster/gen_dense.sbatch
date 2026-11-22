#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 4:00:00
#SBATCH --array=0-95
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/gen_dense_%A_%a.out
# 2026-11-14 dense-frame questions (src/gen_vqa_dense.py), 96 CPU shards (GPFS-bound ~8 s/frame) -> vqa_train_dense_cl
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export PYTHONNOUSERSITE=1 PYTHONPATH=$P/recon/devkit
DENSE_FRAMES=${DENSE_FRAMES:-artifacts/dense_frames_train.json} $P/env/bin/python -u src/gen_vqa_dense.py --out vqa_train_dense_cl --nshards 96 2>&1 | grep --line-buffered -v -E "^  (Clip|Output)|verified all camera"
