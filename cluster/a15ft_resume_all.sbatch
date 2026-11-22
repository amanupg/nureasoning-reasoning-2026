#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH -t 12:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/a15ftall_%j.out
# Continuation for the all-train stage-2 run (runs_a15ft/all): exit 0 if it already reached its
# last step, else resume from the latest checkpoint. Chain several with --dependency=afterany.
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; OUT=$P/runs_a15ft/all
CK=$(ls -d $OUT/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
[ -z "$CK" ] && { echo "no checkpoint"; exit 1; }
DONE=$($P/env/bin/python -c "import json;s=json.load(open(\"$CK/trainer_state.json\"));print(int(s[\"global_step\"]>=s[\"max_steps\"]))")
echo "latest $CK done=$DONE"
[ "$DONE" = 1 ] && exit 0
export HF_HOME=$P/.cache/hf PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 NUREASONING_ROOT=$P PYTHONPATH=$P/src:$P/recon/devkit
export TMPDIR=/dev/shm/$USER_${SLURM_JOB_ID}; mkdir -p $TMPDIR; trap "rm -rf $TMPDIR" EXIT
cd $P/ext/alpamayo-recipes/recipes/alpamayo1_5_sft
$P/ext/a15ftenv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 -m alpamayo1_5_sft.train_hf   --config-path pkg://alpamayo1_5_sft/configs --config-name sft_stage2_nureasoning   data.train_dataset.half=null data.val_dataset.half=null trainer.save_total_limit=2 trainer.save_steps=1000   paths.output_dir=$OUT +resume_from_checkpoint=$CK
