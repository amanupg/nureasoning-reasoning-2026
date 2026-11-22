#!/bin/bash
#SBATCH -A edu
#SBATCH -p burst
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 3-00:00:00
#SBATCH -J vlm_guard
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/vlm_guard_%j.out
# 2026-11-15 (Aman away): CPU-only guard for the two reasoning runs.
# 1) Once the LR-1e-4 LoRA run (vlm_dense_lr1e4) is RUNNING, cancel the LR-2e-4 chain (vlm_dense): its GPUs were kept busy
#    only until the replacement started.
# 2) Each new snapshot (step 500, 1000, ...) of qd4b_dense_lr1e4 / qd4b_dense (LoRA adapters) and qd4b_full (HF dir) gets
#    one cluster/vlm_ckpt_eval.sbatch job (run 1's 4,800 validation questions, paired comparison vs run 1).
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P; mkdir -p runs_vlm/ckpt_eval
while true; do
  if squeue -u "$USER" -h -n vlm_dense_lr1e4 -t R | grep -q . && squeue -u "$USER" -h -n vlm_dense | grep -q .; then
    echo "$(date -u +%FT%T) 1e-4 LoRA running -> cancelling the 2e-4 chain"; squeue -u "$USER" -h -n vlm_dense -o "%i %T"; scancel -u "$USER" -n vlm_dense
  fi
  for spec in qd4b_dense_lr1e4:1 qd4b_dense:1 qd4b_full:0; do run=${spec%%:*}; ad=${spec##*:}
    for d in runs_vlm/$run/step_*; do [ -d "$d" ] || continue; s=${d##*_}
      m=runs_vlm/ckpt_eval/.sub_${run}_$s; [ -f $m ] && continue
      f=$( [ $ad = 1 ] && echo $d/adapter_model.safetensors || echo $d/model.safetensors ); [ -f $f ] || continue
      [ $(( $(date +%s) - $(stat -c %Y $f) )) -lt 180 ] && continue          # written at least 3 min ago
      j=$(SNAP=$d ADAPTER=$ad TAG=${run}_s$s sbatch --parsable cluster/vlm_ckpt_eval.sbatch) && touch $m && echo "$(date -u +%FT%T) eval $d -> $j"
    done
  done
  sleep 120
done
