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
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/fm_ens_%j.out
# Future-motion net (2026-11-08): usage sbatch cluster/fm_ens.sbatch <out dir> [fm_net args]
# Resumes per epoch after preemption. First use of a new backbone downloads it (HF token file, never argv).
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export HF_HOME=$P/.cache/hf PYTHONNOUSERSITE=1
export TMPDIR=/dev/shm/$USER_${SLURM_JOB_ID}; mkdir -p $TMPDIR; trap "rm -rf $TMPDIR" EXIT
STAGE_MODELS="img_crops img_crops90" source cluster/stage_env.sh
W=$LE/work; mkdir -p $W; for d in src artifacts runs_fm cache .cache; do ln -s $P/$d $W/$d; done; ln -s $LE/img_crops $W/img_crops; ln -s $LE/img_crops90 $W/img_crops90; cd $W
$PY -u src/fm_net.py "$@" 2>&1 | grep --line-buffered -v -i warn
