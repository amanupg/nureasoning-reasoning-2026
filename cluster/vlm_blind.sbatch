#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --exclude=ins058,ins082,ins084,ins046,ins047
#SBATCH -t 3:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/vlm_blind_%A_%a.out
# F8 text-reliance diagnostic (2026-11-06): VLM with images replaced by gray ('gray') or by another clip's images
# ('swap'). Validation (held out from run 1, cap 150 per type) with run 1; all 3,000 test questions with run 2.
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export HF_HOME=$P/.cache/hf PYTHONNOUSERSITE=1 PYTHONPATH=$P/recon/devkit
STAGE_CLIPS=16 source cluster/stage_img.sh validation,test
export HF_HUB_OFFLINE=1
STAGE_MODELS="models/qwen_drive_vlm" source cluster/stage_env.sh
M=$LM/models/qwen_drive_vlm
i=$SLURM_ARRAY_TASK_ID
for b in gray swap; do
  $PY -u src/vlm_infer.py validation $M runs_vlm/qd4b_s3000_val150_$b --adapter $P/runs_vlm/qd4b_r64/step_3000 --shard $i --nshards 4 --cap 150 --blind $b
  $PY -u src/vlm_infer.py test $M runs_vlm/qd4b_tv2750_test_$b --adapter $P/runs_vlm/qd4b_tv/step_2750 --shard $i --nshards 4 --blind $b
done 2>&1 | grep --line-buffered -E "questions in shard|done|Error|error"
wc -l runs_vlm/*_gray_s*of4.jsonl runs_vlm/*_swap_s*of4.jsonl
