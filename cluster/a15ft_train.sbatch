#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J a15ft
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=300G
#SBATCH --exclude=ins082
#SBATCH -t 12:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/a15fttrain_%j.out
# usage: sbatch --gres=gpu:N src/a15ft_train.sbatch <hydra overrides...>
source /etc/profile
P=/path/to/cluster/scratch/nureasoning
export HF_HOME=$P/.cache/hf PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 NUREASONING_ROOT=$P
export PYTHONPATH=$P/src:$P/recon/devkit
export TMPDIR=/dev/shm/$USER_${SLURM_JOB_ID}; mkdir -p $TMPDIR; trap "rm -rf $TMPDIR" EXIT   # see nuvla_full.sbatch
cd $P/ext/alpamayo-recipes/recipes/alpamayo1_5_sft
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || exit 3
NG=$(nvidia-smi -L | wc -l)
$P/ext/a15ftenv/bin/python -m torch.distributed.run --standalone --nproc_per_node=$NG -m alpamayo1_5_sft.train_hf   --config-path pkg://alpamayo1_5_sft/configs --config-name ${CONFIG:-sft_stage2_nureasoning} "$@"
