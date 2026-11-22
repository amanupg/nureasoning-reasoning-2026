#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -N 1
#SBATCH -c 10
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ins058,ins082,ins084,ins046,ins047
#SBATCH -t 12:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/dist_ens_%j.out
# Distance-net ensemble members (2026-11-06): usage sbatch cluster/dist_ens.sbatch <out dir> [dist_net args]
# Resumes per epoch after preemption. First use of a new backbone downloads it (HF token file, never argv).
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export HF_HOME=$P/.cache/hf PYTHONNOUSERSITE=1
export TMPDIR=/dev/shm/$USER_${SLURM_JOB_ID}; mkdir -p $TMPDIR; trap "rm -rf $TMPDIR" EXIT
C=${CROPS:-img_crops}; STAGE_MODELS="$C" source cluster/stage_env.sh   # CROPS: alternative crop dir (same ids)
W=$LE/work; mkdir -p $W; for d in src artifacts runs_dist .cache; do ln -s $P/$d $W/$d; done; ln -s $LE/$C $W/img_crops; cd $W
$PY -u src/${SCRIPT:-dist_net.py} "$@" 2>&1 | grep --line-buffered -v -i warn
