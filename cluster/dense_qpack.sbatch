#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 2:00:00
#SBATCH --exclude=ins046,ins047,ins058,ins082,ins084
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/dense_qpack_%j.out
# 2026-11-15 pack dense-run questions into artifacts/dense_q_train.jsonl (src/dense_qpack.py); run after generation.
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export PYTHONNOUSERSITE=1
env/bin/python -u src/dense_qpack.py
