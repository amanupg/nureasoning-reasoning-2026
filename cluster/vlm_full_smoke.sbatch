#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH --gres=gpu:A6000:6
#SBATCH --exclude=ins046,ins047,ins058,ins082,ins084
#SBATCH -t 2:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/vlm_full_smoke_%j.out
# 2026-11-15 smoke of src/vlm_sft_full.py (6 GPUs: 2 OOM at the first Adam step, 13915006): phase 1 = 12 steps (save 10, state 5); phase 2 = resume to 14 in a new
# process; then vlm_infer --limit 20 on the step_10 snapshot (no adapter).
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export HF_HOME=$P/.cache/hf HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$P/recon/devkit DENSE=1
export TMPDIR=/dev/shm/$USER_${SLURM_JOB_ID}; mkdir -p $TMPDIR; trap "rm -rf $TMPDIR" EXIT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 PREFETCH_THREADS=3 IMG_ROOT=$P/img_vlm
MODEL=models/qwen_drive_vlm; STAGE_MODELS="$MODEL" source cluster/stage_env.sh
M=$LM/$MODEL; OUT=$P/runs_vlm/full_smoke; rm -rf $OUT; mkdir -p $OUT
run() { $PY -m torch.distributed.run --standalone --nproc_per_node $(nvidia-smi -L | wc -l) src/vlm_sft_full.py --model $M --out $OUT --steps $1 \
    --accum 2 --lr 1e-5 --warmup 3 --save_every 10 --state_every 5 --init_adapter $P/runs_vlm/qd4b_r64/step_3000 2>&1 \
    | grep --line-buffered -v -i -E "warn|processor_kwargs"; }
echo "=== phase 1 (12 steps)"; run 12
echo "=== phase 2 (resume to 14)"; run 14
echo "=== snapshot"; ls -la $OUT/step_10 $OUT/last
echo "=== infer step_10"
CUDA_VISIBLE_DEVICES=0 $PY -u src/vlm_infer.py validation $OUT/step_10 $OUT/inf --limit 20 2>&1 | grep -E "questions in shard|done|rror|Traceback"
$PY -c "import json; R=[json.loads(l) for l in open('$OUT/inf_s0of1.jsonl')]; print('[infer] n', len(R), 'acc', round(sum(r['pred']==r['gold'] for r in R)/len(R),3))"
echo SMOKE_END
