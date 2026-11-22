#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH --requeue
#SBATCH -N 1
#SBATCH -c 3
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ins046,ins047,ins058,ins082,ins090
#SBATCH -t 4:30:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/vlm_insure_%A_%a.out
# 2026-11-17 insurance while both final runs wait for GPUs: fusion-ready probabilities for a fallback checkpoint
# (the fusion questions, artifacts/fusion_val_qids.txt, and all 3,000 test questions; the cap-300 validation set is
# already in runs_vlm/ckpt_eval/<TAG>). One GPU per array task = one shard (small slots backfill).
# usage: SNAP=runs_vlm/qd4b_dense_lr1e4/step_2500 ADAPTER=1 TAG=qd4b_dense_lr1e4_s2500 sbatch --array=0-1 cluster/vlm_insure.sbatch
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export HF_HOME=$P/.cache/hf HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$P/recon/devkit IMG_ROOT=$P/img_vlm
export TMPDIR=/dev/shm/$USER_${SLURM_JOB_ID}; mkdir -p $TMPDIR; trap "rm -rf $TMPDIR" EXIT
if [ "${ADAPTER:-0}" = 1 ]; then M=$P/models/qwen_drive_vlm; AD="--adapter $P/$SNAP"; else M=$P/$SNAP; AD=""; fi
i=$SLURM_ARRAY_TASK_ID; n=${SLURM_ARRAY_TASK_COUNT:-2}; O=$P/runs_vlm/ckpt_eval/$TAG
env/bin/python -u src/vlm_infer.py validation $M ${O}_valfus $AD --qids artifacts/fusion_val_qids.txt --shard $i --nshards $n 2>&1 | grep --line-buffered -E "questions in shard|done|rror"
env/bin/python -u src/vlm_infer.py test $M ${O}_test $AD --shard $i --nshards $n 2>&1 | grep --line-buffered -E "questions in shard|done|rror"
wc -l ${O}_valfus_s${i}of$n.jsonl ${O}_test_s${i}of$n.jsonl; echo INSURE_DONE
