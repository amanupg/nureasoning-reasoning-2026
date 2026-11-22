#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=50G
#SBATCH --gres=gpu:2
#SBATCH --exclude=ins046,ins047,ins058,ins082,ins084,ins090
#SBATCH -t 12:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/vlm_c3_%j.out
# 2026-11-13 reasoning push: LoRA SFT of a new VLM base (Cosmos3 understanding tower as Qwen3-VL) on train questions
# (src/vlm_sft.py, unchanged recipe: 8 cams x 2 frames, 1-3 questions per draw, test-mix type weights), then letter
# probabilities on held-out validation (vqa_val_cl, cap 300 per type) and on all test questions.
# Self-resuming (state every 25 steps); requeue-safe. Small CPU ask: A6000 nodes are CPU-saturated by other users.
# usage: MODEL=models/cosmos3_nano_vlm RUN=c3n_r64 STEPS=1000 ACCUM=8 sbatch cluster/vlm_c3.sbatch
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export HF_HOME=$P/.cache/hf HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$P/recon/devkit
export TMPDIR=/dev/shm/$USER_${SLURM_JOB_ID}; mkdir -p $TMPDIR; trap "rm -rf $TMPDIR" EXIT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 PREFETCH_THREADS=${PREFETCH_THREADS:-3}
NG=$(nvidia-smi -L | wc -l); GPR=${GPR:-1}; NR=$((NG / GPR))   # GPR>1: layer-split copies (Super)
export GPUS_PER_RANK=$GPR
if [ -d $P/img_vlm ]; then export IMG_ROOT=$P/img_vlm; echo "[stage] shared frames $IMG_ROOT"   # persisted copy (cluster/persist_img.sbatch)
else STAGE_CLIPS=8 source cluster/stage_img.sh train,validation,test; fi
STAGE_MODELS="$MODEL" source cluster/stage_env.sh
M=$LM/$MODEL; OUT=$P/runs_vlm/$RUN; mkdir -p $OUT
[ -f $OUT/c3_done ] && { echo "[skip] $RUN already finished"; echo C3_DONE; exit 0; }
if [ ! -f $OUT/zs_done ]; then          # zero-shot sanity on 160 validation questions (10 per type) before training
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPR-1))) $PY -u src/vlm_infer.py validation $M $OUT/zs --cap 10 2>&1 | grep --line-buffered -E "questions in shard|done|rror"
  $PY -c "import json; R=[json.loads(l) for l in open('$OUT/zs_s0of1.jsonl')]; print('[zs] zero-shot val acc', round(sum(r['pred']==r['gold'] for r in R)/len(R),3), 'n', len(R))"
  touch $OUT/zs_done
fi
$PY -m torch.distributed.run --standalone --nproc_per_node $NR src/vlm_sft.py --model $M --out $OUT --steps ${STEPS:-1000} \
    --accum ${ACCUM:-8} --lr ${LR:-1e-4} --rank ${RANK_LORA:-64} --warmup ${WARMUP:-50} --save_every ${SAVE:-250} --state_every 25 --split train ${INIT:+--init_adapter $P/$INIT} 2>&1 \
    | grep --line-buffered -v -i -E "warn|processor_kwargs"
A=$OUT/step_${STEPS:-1000}
[ -d $A ] || { echo "no final adapter"; exit 1; }
for i in $(seq 0 $((NR-1))); do
  G=$(seq -s, $((i*GPR)) $((i*GPR+GPR-1)))
  ( CUDA_VISIBLE_DEVICES=$G $PY -u src/vlm_infer.py validation $M $OUT/val --adapter $A --cap 300 --shard $i --nshards $NR
    CUDA_VISIBLE_DEVICES=$G $PY -u src/vlm_infer.py test $M $OUT/test --adapter $A --shard $i --nshards $NR
  ) 2>&1 | grep --line-buffered -E "questions in shard|done|rror" | sed "s/^/[gpu$i] /" &
done
wait
cat $OUT/val_s*of$NR.jsonl | wc -l; cat $OUT/test_s*of$NR.jsonl | wc -l; touch $OUT/c3_done; echo C3_DONE
