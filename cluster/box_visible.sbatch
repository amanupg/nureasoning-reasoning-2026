#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ins058,ins082,ins084,ins046,ins047
#SBATCH -t 4:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/box_visible_%j.out
# usage: sbatch cluster/box_visible.sbatch <split> <out.jsonl> [extra args]
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export HF_HOME=$P/.cache/hf HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$P/recon/devkit
env/bin/python -u src/box_visible.py "$@" 2>&1 | grep --line-buffered -v -i -E "warn"
