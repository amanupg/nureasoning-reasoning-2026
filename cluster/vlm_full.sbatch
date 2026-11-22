#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH --gres=gpu:A6000:6
#SBATCH --exclude=ins046,ins047,ins058,ins082,ins084,ins090
#SBATCH -t 12:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/vlm_full_%j.out
# 2026-11-15 limitations audit: FULL fine-tune (FSDP2, every weight) of the Qwen-Drive VLM, warm start = run 1 merged
# (src/vlm_sft_full.py), same data as the LoRA runs (DENSE=1: dense (clip, frame) examples). Then letter probabilities
# on held-out validation (vqa_val_cl, cap 300 per type) and on all test questions, from the final HF snapshot.
# Self-resuming (per-rank shard state every STATE=50 steps: 6 x 9 GB, about 1 min); requeue-safe; chain copies with -J <name> --dependency=singleton.
# warmup defaults to 5% of STEPS, weight decay 0.1 (leader full-FT recipe).
# usage: DENSE=1 MODEL=models/qwen_drive_vlm RUN=qd4b_full STEPS=3000 ACCUM=5 LR=1e-5 INIT=runs_vlm/qd4b_r64/step_3000 sbatch cluster/vlm_full.sbatch
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export HF_HOME=$P/.cache/hf HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$P/recon/devkit
export TMPDIR=/dev/shm/$USER_${SLURM_JOB_ID}; mkdir -p $TMPDIR; trap "rm -rf $TMPDIR" EXIT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 PREFETCH_THREADS=${PREFETCH_THREADS:-3}
NR=$(nvidia-smi -L | wc -l)
export IMG_ROOT=$P/img_vlm; echo "[stage] shared frames $IMG_ROOT"
STAGE_MODELS="$MODEL" source cluster/stage_env.sh
M=$LM/$MODEL; OUT=$P/runs_vlm/$RUN; mkdir -p $OUT
[ -f $OUT/full_done ] && { echo "[skip] $RUN already finished"; echo FULL_DONE; exit 0; }
$PY -m torch.distributed.run --standalone --nproc_per_node $NR src/vlm_sft_full.py --model $M --out $OUT --steps ${STEPS:-3000} \
    --accum ${ACCUM:-5} --lr ${LR:-1e-5} --wd ${WD:-0.1} --warmup ${WARMUP:-$(( ${STEPS:-3000} / 20 ))} --save_every ${SAVE:-500} --state_every ${STATE:-50} \
    --split train ${INIT:+--init_adapter $P/$INIT} 2>&1 | grep --line-buffered -v -i -E "warn|processor_kwargs"
A=$OUT/step_${STEPS:-3000}
[ -f $A/model.safetensors ] || { echo "no final snapshot"; exit 1; }
[ "${NOINFER:-0}" = 1 ] && { echo "NOINFER set: skipping inference"; exit 0; }
FM=$LM/final_model; cp -r $A $FM                      # one sequential read to node-local disk
for i in $(seq 0 $((NR-1))); do
  ( CUDA_VISIBLE_DEVICES=$i $PY -u src/vlm_infer.py validation $FM $OUT/val --cap 300 --shard $i --nshards $NR
    CUDA_VISIBLE_DEVICES=$i $PY -u src/vlm_infer.py test $FM $OUT/test --shard $i --nshards $NR
  ) 2>&1 | grep --line-buffered -E "questions in shard|done|rror" | sed "s/^/[gpu$i] /" &
done
wait
cat $OUT/val_s*of$NR.jsonl | wc -l; cat $OUT/test_s*of$NR.jsonl | wc -l; touch $OUT/full_done; echo FULL_DONE
