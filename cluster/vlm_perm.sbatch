#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --exclude=ins058,ins082,ins084,ins046,ins047
#SBATCH -t 5:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/vlm_perm_%j.out
# Option-order debiasing: VLM runs 1 and 2 on the VLM-routed test types with the options rotated by k = 1, 2, 3
# (k = 0 = the existing outputs). Test frames are staged once to node-local disk; 4 GPUs, one shard each.
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export HF_HOME=$P/.cache/hf PYTHONNOUSERSITE=1 PYTHONPATH=$P/recon/devkit
STAGE_CLIPS=16 source cluster/stage_img.sh test
export HF_HUB_OFFLINE=1
SUBS=image_region_from_3d_object,closest_object_type,motion_relation,action_risk_assessment,unsafe_action_identification,driving_decision_joint,driving_decision_lateral
M=$P/models/qwen_drive_vlm
for i in 0 1 2 3; do
  ( for k in 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i env/bin/python -u src/vlm_infer.py test $M runs_vlm/qd4b_tv2750_test_p$k --adapter $P/runs_vlm/qd4b_tv/step_2750 --shard $i --nshards 4 --subs $SUBS --perm $k
      CUDA_VISIBLE_DEVICES=$i env/bin/python -u src/vlm_infer.py test $M runs_vlm/qd4b_s3000_test_p$k --adapter $P/runs_vlm/qd4b_r64/step_3000 --shard $i --nshards 4 --subs $SUBS --perm $k
    done ) 2>&1 | grep --line-buffered -E "questions in shard|done|Error|error" | sed "s/^/[gpu$i] /" &
done
wait
ls -la runs_vlm/*_test_p*_s*of4.jsonl | wc -l
