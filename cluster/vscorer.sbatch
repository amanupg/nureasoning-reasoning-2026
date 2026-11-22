#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J vscorer
#SBATCH --gres=gpu:1
#SBATCH --exclude=ins082
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/vscorer_%j.out
source /etc/profile
export HF_HOME=/path/to/cluster/scratch/nureasoning/.cache/hf PYTHONNOUSERSITE=1
cd /path/to/cluster/scratch/nureasoning
nvidia-smi --query-gpu=name --format=csv,noheader || exit 3
/path/to/cluster/scratch/nureasoning/env/bin/python -u src/vscorer.py $ARGS
