#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/img_%x_%A_%a.out
source /etc/profile
export HF_HOME=/path/to/cluster/scratch/nureasoning/.cache/hf PYTHONNOUSERSITE=1
cd /path/to/cluster/scratch/nureasoning
/path/to/cluster/scratch/nureasoning/env/bin/python src/img_ingest.py --members src/image_members.json   --split $SPLIT --out img --shard $SLURM_ARRAY_TASK_ID --nshards $NSHARDS --workers 8
