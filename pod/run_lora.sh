#!/bin/bash
# 2026-11-16 dense LoRA continued on a rented 8-GPU pod: warm start from the Insomnia run's latest adapter
# (runs_vlm/qd4b_dense_lr1e4/step_N, copied to init_adapter/), same recipe as on Insomnia (LoRA r64 incl. vision,
# LR 1e-4 peak, cosine, 32 examples/step = 8 ranks x accum 4, k 1/12 .1/.9), data = key + passes A/B/C (126K frames).
# Fresh AdamW + warmup 150 (optimizer state not carried over; world size differs).
# usage: STEPS=<from SPEEDTEST> bash pod/run_lora.sh         speed probe first: SPEEDTEST=1 bash pod/run_lora.sh
set -e; cd /workspace/nureasoning; mkdir -p logs
export DENSE=1 DENSE_PASSES=ABC DENSE_INDEX=artifacts/dense_index_train_v2.json DENSE_QPACK=artifacts/dense_q_train_v2.jsonl \
  DENSE_KS=1,12 DENSE_KW=.1,.9 IMG_ROOT=img_vlm VISION=lora PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=2 PREFETCH_THREADS=${PREFETCH_THREADS:-4} HF_HUB_OFFLINE=1 GPU_MEM=${GPU_MEM:-70GiB}
NR=$(nvidia-smi -L | wc -l); ACC=$(( 32 / NR ))
if [ -n "$SPEEDTEST" ]; then RUN=speedtest; STEPS=40; rm -rf runs_vlm/speedtest; fi
torchrun --standalone --nproc_per_node $NR src/vlm_sft.py --model models/qwen_drive_vlm --out runs_vlm/${RUN:-qd4b_dense_pod} \
  --steps ${STEPS:?} --accum $ACC --lr ${LR:-1e-4} --rank 64 --warmup ${WARMUP:-150} --save_every ${SAVE:-500} \
  --state_every ${STATE:-50} --split train --init_adapter init_adapter 2>&1 \
  | grep --line-buffered -v -i -E "warn|processor_kwargs" | tee -a logs/${RUN:-qd4b_dense_pod}.log
