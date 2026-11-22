#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J a15ftres
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --exclude=ins082
#SBATCH -t 6:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/a15ftres_%j.out
# Safety continuation: exit 0 if the run already reached its last step, else resume.
# usage: sbatch src/a15ft_resume.sbatch <half> <other-half>
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; H=$1; O=$2; OUT=$P/runs_a15ft/$H
CK=$(ls -d $OUT/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
[ -z "$CK" ] && { echo "no checkpoint"; exit 1; }
DONE=$($P/env/bin/python -c "import json;s=json.load(open(\"$CK/trainer_state.json\"));print(int(s[\"global_step\"]>=s[\"max_steps\"]))")
echo "latest $CK done=$DONE"
[ "$DONE" = 1 ] && exit 0
export HF_HOME=$P/.cache/hf PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 NUREASONING_ROOT=$P PYTHONPATH=$P/src:$P/recon/devkit
cd $P/ext/alpamayo-recipes/recipes/alpamayo1_5_sft
$P/ext/a15ftenv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 -m alpamayo1_5_sft.train_hf   --config-path pkg://alpamayo1_5_sft/configs --config-name sft_stage2_nureasoning   data.train_dataset.half=$H data.val_dataset.half=$O trainer.save_total_limit=2 trainer.save_steps=2000   paths.output_dir=$OUT +resume_from_checkpoint=$CK
