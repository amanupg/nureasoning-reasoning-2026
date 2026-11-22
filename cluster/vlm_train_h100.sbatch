#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:2
#SBATCH --exclude=ins058,ins082,ins084,ins046,ins047
#SBATCH -t 12:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/vlm_train_%j.out
# Self-resuming LoRA SFT of the VLM on generated questions (src/vlm_sft.py), 2 H100s.
# usage: sbatch cluster/vlm_train.sbatch <model dir> <out dir> [extra vlm_sft args...]
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export HF_HOME=$P/.cache/hf HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$P/recon/devkit
export TMPDIR=/dev/shm/$USER_${SLURM_JOB_ID}; mkdir -p $TMPDIR; trap "rm -rf $TMPDIR" EXIT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
M=$1; OUT=$2; shift 2
env/bin/torchrun --standalone --nproc_per_node 2 src/vlm_sft.py --model $M --out $OUT "$@" 2>&1 | grep --line-buffered -v -i -E "warn|processor_kwargs"
