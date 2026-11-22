#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J lpaths
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/lpaths_%A_%a.out
source /etc/profile
export PYTHONNOUSERSITE=1
cd /path/to/cluster/scratch/nureasoning
env/bin/python -u src/label_paths.py --split $SPLIT --kind $KIND --src $SRC --tag $TAG --shard $SLURM_ARRAY_TASK_ID --nshards $NSH
