#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J edge_vqa
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ins046,ins047,ins058,ins082,ins084
#SBATCH -t 6:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/edge_vqa_%j.out
# usage: sbatch cluster/edge_vqa.sbatch <split> <out prefix> [edge_vqa args]   (resumes per question)
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export HF_HOME=$P/.cache/hf PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 IMG_ROOT=$P/img_edge
LE=/local/$USER_${SLURM_JOB_ID}_edge; mkdir -p $LE; trap "rm -rf $LE" EXIT
t0=$(date +%s); tar -C $LE -xf ext/edge_env.tar; echo "[env] staged in $(( $(date +%s) - t0 )) s"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
S=$1; O=$2; shift 2
$LE/edge_env/bin/python -u src/edge_vqa.py $S models/alpamayo2_edge $O "$@" 2>&1 | grep --line-buffered -v -i warn
