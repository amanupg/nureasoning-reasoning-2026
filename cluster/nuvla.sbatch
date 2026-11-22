#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J nuvla
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH --exclude=ins082
#SBATCH -t 12:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/nuvla_%j.out
# usage: sbatch --gres=gpu:N src/nuvla.sbatch <nuvla.train args...>
source /etc/profile
export HF_HOME=/path/to/cluster/scratch/nureasoning/.cache/hf PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 NUVLA_REQUIRE_IMAGES=1 NUVLA_ATTN=sdpa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /path/to/cluster/scratch/nureasoning
export PYTHONPATH=$PWD/recon/devkit
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || exit 3
NG=$(nvidia-smi -L | wc -l)
env/bin/python -m torch.distributed.run --standalone --nproc_per_node=$NG -m nureasoning.nuvla.train "$@"
