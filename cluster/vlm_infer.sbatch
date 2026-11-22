#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ins058,ins082,ins084,ins046,ins047
#SBATCH -t 6:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/vlm_infer_%j.out
# VLM letter probabilities, one shard per array task. usage:
#   NSH=4 sbatch --array=0-3 cluster/vlm_infer.sbatch <split> <model> <out prefix> [adapter] [extra args]
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export HF_HOME=$P/.cache/hf HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$P/recon/devkit
export TMPDIR=/dev/shm/$USER_${SLURM_JOB_ID}; mkdir -p $TMPDIR; trap "rm -rf $TMPDIR" EXIT
S=$1; M=$2; O=$3; A=${4:-}; shift 4 2>/dev/null || shift $#
env/bin/python -u src/vlm_infer.py $S $M $O ${A:+--adapter $A} --shard ${SLURM_ARRAY_TASK_ID:-0} --nshards ${NSH:-1} "$@" 2>&1 | grep --line-buffered -v -i -E "warn|processor_kwargs|regex"
