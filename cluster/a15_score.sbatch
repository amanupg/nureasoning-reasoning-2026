#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J a15scr
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/a15scr_%j.out
source /etc/profile
export PYTHONNOUSERSITE=1
cd /path/to/cluster/scratch/nureasoning
for v in nonav nav; do env/bin/python -u src/a15_score.py --variant $v; done
