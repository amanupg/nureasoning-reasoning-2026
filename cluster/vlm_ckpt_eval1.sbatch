#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH --requeue
#SBATCH -N 1
#SBATCH -c 3
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ins046,ins047,ins058,ins082,ins090
#SBATCH -t 3:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/vlm_ckpt_eval1_%A_%a.out
# 2026-11-17: same as vlm_ckpt_eval.sbatch (cap-300 validation, run 1's questions) but one GPU per array task so it
# backfills; the paired comparison (src/vlm_pair_cmp.py) runs after all shards finish.
# usage: SNAP=... ADAPTER=1 TAG=... sbatch --array=0-1 cluster/vlm_ckpt_eval1.sbatch
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export HF_HOME=$P/.cache/hf HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$P/recon/devkit IMG_ROOT=$P/img_vlm
export TMPDIR=/dev/shm/$USER_${SLURM_JOB_ID}; mkdir -p $TMPDIR; trap "rm -rf $TMPDIR" EXIT
if [ "${ADAPTER:-0}" = 1 ]; then M=$P/models/qwen_drive_vlm; AD="--adapter $P/$SNAP"; else M=$P/$SNAP; AD=""; fi
i=$SLURM_ARRAY_TASK_ID; n=${SLURM_ARRAY_TASK_COUNT:-2}; OUT=$P/runs_vlm/ckpt_eval/$TAG; mkdir -p $(dirname $OUT)
env/bin/python -u src/vlm_infer.py validation $M $OUT $AD --cap 300 --shard $i --nshards $n 2>&1 | grep --line-buffered -E "questions in shard|done|rror"
wc -l ${OUT}_s${i}of$n.jsonl; echo EVAL1_DONE
