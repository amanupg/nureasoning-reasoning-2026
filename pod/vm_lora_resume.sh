#!/bin/bash
# 2026-11-17 Nebius 1x B200 (preemptible): continue the Insomnia LoRA run qd4b_dense_lr1e4 from its last state
# (step 2900) to 3500 with the identical recipe and data (v1 dense data, passes A/B + key frames; env as sbatch'd on
# Insomnia: DENSE=1 DENSE_KS=1,12 DENSE_KW=.1,.9 VISION=lora, LR 1e-4, warmup 150, 3500 steps, rank 64).
# 32 examples/step = 1 rank x accum 32 (Insomnia: 4 ranks x 8). Optimizer state + adapter carried over (state.pt).
# Stage 1 (UNPACK=1): verify md5 of every part, unpack, check member counts. Stage 2: train (resumes from last/).
# usage: UNPACK=1 bash pod/vm_lora_resume.sh   then   bash pod/vm_lora_resume.sh
cd ~/w; . ~/env/bin/activate
if [ -n "$UNPACK" ]; then
  md5sum -c code.md5 && tar xf code_artifacts.tar || { echo CODE_BAD; exit 1; }
  ( cd packs && md5sum -c MD5SUMS --quiet ) && ( cd packs2 && md5sum -c MD5SUMS --quiet ) && echo MD5_OK || { echo MD5_BAD; exit 1; }
  for f in packs/models_qwen_drive_vlm.files packs2/*.c*.files; do ( n=${f%.files}; cat $n.tar.part-* | tar xf - ;
    python - "$f" <<'PY'
import os, sys; L = [l.rstrip("\n") for l in open(sys.argv[1])]; got = sum(os.path.isfile(p) for p in L)
print(f"[unpack] {sys.argv[1]}: {got} / {len(L)}", flush=True); sys.exit(got != len(L))
PY
  ) & done; wait; echo UNPACK_DONE; exit 0
fi
export DENSE=1 DENSE_KS=1,12 DENSE_KW=.1,.9 VISION=lora IMG_ROOT=img_vlm HF_HUB_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 PREFETCH_THREADS=${PREFETCH_THREADS:-14} GPU_MEM=170GiB
python -u src/vlm_sft.py --model models/qwen_drive_vlm --out runs_vlm/qd4b_dense_lr1e4 --steps 3500 --accum 32 --lr 1e-4 \
  --rank 64 --warmup 150 --save_every 500 --state_every 25 --split train --init_adapter runs_vlm/qd4b_r64/step_3000 2>&1 \
  | grep --line-buffered -v -i -E "warn|processor_kwargs" | tee -a logs_lora.txt
