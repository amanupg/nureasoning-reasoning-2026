#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:A6000:2
#SBATCH --exclude=ins046,ins047,ins058,ins082,ins084,ins090
#SBATCH -t 2:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/dense_label_check_%j.out
# 2026-11-16: run 1 (never trained on extra frames) on 200 extra-frame questions per type (+ gray-image copies of
# distance / image_region). Near validation accuracy = dense data aligned; near gray = misaligned. src/dense_label_check.py
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export HF_HOME=$P/.cache/hf HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$P/recon/devkit IMG_ROOT=$P/img_vlm
export TMPDIR=/dev/shm/$USER_${SLURM_JOB_ID}; mkdir -p $TMPDIR; trap "rm -rf $TMPDIR" EXIT
OUT=$P/runs_vlm/ckpt_eval/dense_check_${TAG:-run1}; AD=${ADAPTER:-runs_vlm/qd4b_r64/step_3000}
for i in 0 1; do CUDA_VISIBLE_DEVICES=$i env/bin/python -u src/dense_label_check.py $P/models/qwen_drive_vlm $OUT --adapter $P/$AD --shard $i --nshards 2 2>&1 \
  | grep --line-buffered -E "questions in shard|done|rror|Trace" | sed "s/^/[gpu$i] /" & done; wait
env/bin/python - <<PY
import json, glob, collections
R = [json.loads(l) for f in glob.glob("${OUT}_s*of2.jsonl") for l in open(f)]
acc = collections.defaultdict(list)
for r in R: acc[(r["subcategory"], r["blind"] or "img")].append(r["pred"] == r["gold"])
for k in sorted(acc): print(f"{k[0]:32s} {k[1]:5s} n {len(acc[k]):4d} acc {sum(acc[k]) / len(acc[k]):.3f}")
PY
echo CHECK_DONE
