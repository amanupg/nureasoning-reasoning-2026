#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ins082,ins084,ins046,ins047
#SBATCH -t 6:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/fm_track_%j.out
# SAM video tracking of future_motion boxes. usage: sbatch [--array=0-(n-1)] cluster/fm_track.sbatch <split> <out_prefix> <model> [roots]
# One shard per array task (n = NSH, default 1).
source /etc/profile
P=/path/to/cluster/scratch/nureasoning
export HF_HOME=$P/.cache/hf HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$P/recon/devkit
export TMPDIR=/dev/shm/$USER_${SLURM_JOB_ID}; mkdir -p $TMPDIR; trap "rm -rf $TMPDIR" EXIT
cd $P
env/bin/python -u src/fm_track.py $1 $2 --model $3 --roots ${4:-img_fm:img_a15:img} --shard ${SLURM_ARRAY_TASK_ID:-0} --nshards ${NSH:-1} --first ${FIRST:-90} --sub ${SUB:-future_motion_label}
