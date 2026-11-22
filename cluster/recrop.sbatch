#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/recrop_%j.out
# 2026-11-14: wider-context distance crops from img_vlm key frames (src/recrop.py). usage: sbatch cluster/recrop.sbatch <out> <K>
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
env/bin/python -u src/recrop.py $1 --k $2 --workers 16
