#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J a15inf
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ins082
#SBATCH -t 6:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/a15inf_%j.out
source /etc/profile
export HF_HOME=/path/to/cluster/scratch/nureasoning/.cache/hf PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
cd /path/to/cluster/scratch/nureasoning
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || exit 1
ext/a15env/bin/python -u src/a15_infer.py "$@"
