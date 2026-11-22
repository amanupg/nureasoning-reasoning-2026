#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J cache
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/cache_%j.out
source /etc/profile
export HF_HOME=/path/to/cluster/scratch/nureasoning/.cache/hf PYTHONNOUSERSITE=1
cd /path/to/cluster/scratch/nureasoning
for S in test validation train; do
  /path/to/cluster/scratch/nureasoning/env/bin/python -u src/ingest.py --split $S --tree recon/tree_full.json --workers 8 --tail-mb $([ $S = test ] && echo 2 || echo 20)
done
