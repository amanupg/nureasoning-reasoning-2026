#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:A6000:2
#SBATCH --exclude=ins046,ins047,ins058,ins082,ins084,ins090
#SBATCH -t 4:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/vlm_ckpt_eval_%j.out
# 2026-11-15: validation check of one training snapshot on run 1's exact 1,602 questions (vlm_infer --cap 300, each
# question alone), then the paired per-type comparison vs run 1 (src/vlm_pair_cmp.py, bootstrap by clip).
# usage: SNAP=runs_vlm/qd4b_dense/step_500 [ADAPTER=1] TAG=dense_s500 sbatch cluster/vlm_ckpt_eval.sbatch
#   ADAPTER=1: SNAP is a LoRA adapter on models/qwen_drive_vlm; else SNAP is a full HF model dir (full fine-tune)
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export HF_HOME=$P/.cache/hf HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$P/recon/devkit IMG_ROOT=$P/img_vlm
export TMPDIR=/dev/shm/$USER_${SLURM_JOB_ID}; mkdir -p $TMPDIR; trap "rm -rf $TMPDIR" EXIT
OUT=$P/runs_vlm/ckpt_eval/$TAG; mkdir -p $(dirname $OUT)
if [ "${ADAPTER:-0}" = 1 ]; then M=$P/models/qwen_drive_vlm; AD="--adapter $P/$SNAP"; else M=$P/$SNAP; AD=""; fi
for i in 0 1; do CUDA_VISIBLE_DEVICES=$i env/bin/python -u src/vlm_infer.py validation $M $OUT $AD --cap 300 --shard $i --nshards 2 2>&1 \
  | grep --line-buffered -E "questions in shard|done|rror" | sed "s/^/[gpu$i] /" & done; wait
cat ${OUT}_s*of2.jsonl | wc -l
env/bin/python src/vlm_pair_cmp.py "runs_vlm/qd4b_s3000_val_s*of4.jsonl" "${OUT}_s*of2.jsonl"
echo EVAL_DONE
