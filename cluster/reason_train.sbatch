#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --exclude=ins082,ins046,ins047
#SBATCH -t 12:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/reasontrain_%j.out
# Self-resuming devkit reasoning LoRA SFT. usage: sbatch src/reason_train.sbatch <config.yaml> <output_dir>
# Exits 0 if the latest checkpoint already reached max_steps; else resumes from it (or starts).
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; CFG=$1; OUT=$2
export HF_HOME=$P/.cache/hf HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$P/recon/devkit
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/dev/shm/$USER_${SLURM_JOB_ID}; mkdir -p $TMPDIR; trap "rm -rf $TMPDIR" EXIT
cd $P
CK=$(ls -d $OUT/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
RES=""
if [ -n "$CK" ]; then
  DONE=$(env/bin/python -c "import json;s=json.load(open(\"$CK/trainer_state.json\"));print(int(s[\"global_step\"]>=s[\"max_steps\"]))")
  echo "latest $CK done=$DONE"; [ "$DONE" = 1 ] && exit 0
  RES="--resume-from-checkpoint $CK"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
env/bin/python -m torch.distributed.run --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) -m nureasoning.reasoning.train --config $CFG $RES
