#!/bin/bash
# 2026-11-16 dense full fine-tune continued on a rented 8-GPU pod. INIT = an HF snapshot dir from Insomnia (bf16, loads
# with no adapter); fresh AdamW with warmup (optimizer state is not carried over). Data = key + passes A/B/C (126K frames).
# usage: INIT=init_model STEPS=12000 LR=5e-6 bash pod/run_full.sh
set -e; cd /workspace/nureasoning
export DENSE=1 DENSE_PASSES=ABC DENSE_INDEX=artifacts/dense_index_train_v2.json DENSE_QPACK=artifacts/dense_q_train_v2.jsonl \
  DENSE_KS=1,12 DENSE_KW=.1,.9 IMG_ROOT=img_vlm PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 \
  PREFETCH_THREADS=${PREFETCH_THREADS:-4} HF_HUB_OFFLINE=1
NR=$(nvidia-smi -L | wc -l)
torchrun --standalone --nproc_per_node $NR src/vlm_sft_full.py --model ${INIT:-init_model} --out runs_vlm/${RUN:-qd4b_full_pod} \
  --steps ${STEPS:?} --accum ${ACCUM:-4} --lr ${LR:-5e-6} --wd 0.1 --warmup ${WARMUP:-200} --save_every ${SAVE:-1000} \
  --state_every ${STATE:-100} --split train 2>&1 | grep --line-buffered -v -i -E "warn|processor_kwargs" | tee -a logs/full_pod.log
