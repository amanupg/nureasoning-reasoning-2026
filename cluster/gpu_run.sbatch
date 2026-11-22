#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J gpurun
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ins082,ins046,ins047
#SBATCH -t 2:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/gpurun_%j.out
# usage: sbatch src/gpu_run.sbatch -m module args...
source /etc/profile
export HF_HOME=/path/to/cluster/scratch/nureasoning/.cache/hf PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 NUVLA_ATTN=sdpa
cd /path/to/cluster/scratch/nureasoning
export PYTHONPATH=$PWD/recon/devkit
nvidia-smi --query-gpu=name --format=csv,noheader || exit 3
env/bin/python "$@"
