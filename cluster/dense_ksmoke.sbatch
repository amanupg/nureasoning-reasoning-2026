#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH --gres=gpu:A6000:1
#SBATCH --exclude=ins046,ins047,ins058,ins082,ins084
#SBATCH -t 1:30:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/dense_ksmoke_%j.out
# 2026-11-15: questions-per-example dose test for the dense LoRA run. Realised k, then 10 steps at k=12 and at k=6
# (accum 2, 1 A6000): peak GPU memory (nvidia-smi poll) and s/step for each. Loss must stay finite.
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export HF_HOME=$P/.cache/hf HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$P/recon/devkit
export TMPDIR=/dev/shm/$USER_${SLURM_JOB_ID}; mkdir -p $TMPDIR; trap "rm -rf $TMPDIR; kill %1 2>/dev/null" EXIT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 IMG_ROOT=$P/img_vlm DENSE=1 VISION=lora
for K in "12:1" "1,6,12:.15,.35,.5" "6:1"; do
  export DENSE_KS=${K%%:*} DENSE_KW=${K##*:}
  env/bin/python -c "
import sys; sys.path.insert(0,'src'); from collections import Counter; import vlm_data_dense as d
t=d.TrainSet('train'); n=[len(t.draw()[1]) for _ in range(2000)]; print('[k] KS', d.KS, 'KW', d.KW, 'realised mean k', sum(n)/len(n), sorted(Counter(n).items()))" 2>&1 | grep "^\[k\]"
done
for K in 12 6; do
  export DENSE_KS=$K DENSE_KW=1; rm -rf runs_vlm/ksmoke_$K
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > $TMPDIR/mem_$K.txt &
  env/bin/python -u src/vlm_sft.py --model models/qwen_drive_vlm --out runs_vlm/ksmoke_$K --steps 20 --accum 2 --lr 1e-4 \
    --rank 64 --warmup 2 --save_every 1000 --state_every 1000 --split train --init_adapter $P/runs_vlm/qd4b_r64/step_3000 2>&1 \
    | grep --line-buffered -E "^step|rror|Traceback|OutOfMemory"
  kill %1; echo "[mem] k=$K peak MiB $(sort -n $TMPDIR/mem_$K.txt | tail -1)"; rm -rf runs_vlm/ksmoke_$K
done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo KSMOKE_DONE
