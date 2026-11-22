#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2
#SBATCH --exclude=ins058,ins082,ins084,ins046,ins047
#SBATCH -t 12:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/vlm_meta_%j.out
# Ego meta-action (future-derived stop/decel/keep/accel, src/meta_labels.py) as a VLM question, for the planning
# thread (2026-11-06). Warm start from VLM run 1 (train only; run 2 saw validation decision labels), train split only,
# META_P of draws include the clip's meta question. Then letter probabilities on all validation and test clips.
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export HF_HOME=$P/.cache/hf HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$P/recon/devkit
export TMPDIR=/dev/shm/$USER_${SLURM_JOB_ID}; mkdir -p $TMPDIR; trap "rm -rf $TMPDIR" EXIT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
STAGE_CLIPS=16 source cluster/stage_img.sh train,validation,test
STAGE_MODELS="models/qwen_drive_vlm" source cluster/stage_env.sh
export META_Q=1 META_P=${META_P:-0.6}
M=$LM/models/qwen_drive_vlm; OUT=$P/runs_vlm/${RUN:-meta_r1}
$PY -m torch.distributed.run --standalone --nproc_per_node 2 src/vlm_sft.py --model $M --out $OUT --steps ${STEPS:-900} --accum 8 \
    --lr 5e-5 --warmup 50 --save_every 300 --state_every 50 --split train --init_adapter $P/runs_vlm/qd4b_r64/step_3000 2>&1 \
    | grep --line-buffered -v -i -E "warn|processor_kwargs"
A=$OUT/step_${STEPS:-900}
for i in 0 1; do
  ( META_ALL_CLIPS=1 CUDA_VISIBLE_DEVICES=$i $PY -u src/vlm_infer.py validation $M $OUT/val --adapter $A --shard $i --nshards 2 --subs ego_meta_action
    CUDA_VISIBLE_DEVICES=$i $PY -u src/vlm_infer.py test $M $OUT/test --adapter $A --shard $i --nshards 2 --subs ego_meta_action
  ) 2>&1 | grep --line-buffered -E "questions in shard|done|Error|error" | sed "s/^/[gpu$i] /" &
done
wait
cat $OUT/val_s*of2.jsonl | wc -l; cat $OUT/test_s*of2.jsonl | wc -l
