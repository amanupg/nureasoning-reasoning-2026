#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/l148_%x_%A_%a.out
source /etc/profile
export PYTHONNOUSERSITE=1
cd /path/to/cluster/scratch/nureasoning
/path/to/cluster/scratch/nureasoning/env/bin/python -u src/label148.py --split $SPLIT --nshards $NSH
