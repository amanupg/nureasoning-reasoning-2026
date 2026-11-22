#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ins058,ins082,ins084,ins046,ins047
#SBATCH -t 8:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/dist_net_%j.out
# End-to-end distance regressor on full-resolution crops. usage: sbatch cluster/dist_net.sbatch <out dir> [args]
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export HF_HOME=$P/.cache/hf HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
export TMPDIR=/dev/shm/$USER_${SLURM_JOB_ID}; mkdir -p $TMPDIR; trap "rm -rf $TMPDIR" EXIT
env/bin/python -u src/dist_net.py "$@" 2>&1 | grep --line-buffered -v -i warn
