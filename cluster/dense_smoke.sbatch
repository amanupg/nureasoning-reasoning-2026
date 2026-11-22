#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ins046,ins047,ins058,ins082,ins084
#SBATCH -t 1:30:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/dense_smoke_%j.out
# 2026-11-15 dense-frame run smoke: loader mix over 2,000 draws (vlm_data_dense __main__), then 20 LoRA steps
# (DENSE=1, VISION=lora, warm start from run 1) on whatever dense frames have arrived so far. Loss must fall.
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export HF_HOME=$P/.cache/hf HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$P/recon/devkit
export TMPDIR=/dev/shm/$USER_${SLURM_JOB_ID}; mkdir -p $TMPDIR; trap "rm -rf $TMPDIR" EXIT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 IMG_ROOT=$P/img_vlm DENSE=1 VISION=lora
env/bin/python -u src/vlm_data_dense.py 2>&1 | grep -v -i warn
rm -rf runs_vlm/dense_smoke
env/bin/python -u src/vlm_sft.py --model models/qwen_drive_vlm --out runs_vlm/dense_smoke --steps 20 --accum 2 --lr 1e-4 \
  --rank 64 --warmup 2 --save_every 20 --state_every 1000 --split train --init_adapter $P/runs_vlm/qd4b_r64/step_3000 2>&1 \
  | grep --line-buffered -v -i -E "warn|processor_kwargs"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo DENSE_SMOKE_DONE
