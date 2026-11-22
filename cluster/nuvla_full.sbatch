#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J nuvlafull
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=400G
#SBATCH --gres=gpu:8
#SBATCH --exclude=ins082
#SBATCH -t 12:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/nuvlafull_%j.out
# Self-resuming full-recipe nuVLA: picks up the latest epoch_* checkpoint in OUT.
source /etc/profile
export HF_HOME=/path/to/cluster/scratch/nureasoning/.cache/hf PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export NUVLA_REQUIRE_IMAGES=1 NUVLA_ATTN=sdpa NUVLA_LAZY_META=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# DataLoader workers hand tensors over a unix socket under $TMPDIR (pymp-*); a /tmp cleaner
# removed it ~6 h into two runs ("Pin memory thread exited", FileNotFoundError in SocketClient).
export TMPDIR=/dev/shm/$USER_${SLURM_JOB_ID}; mkdir -p $TMPDIR; trap "rm -rf $TMPDIR" EXIT
cd /path/to/cluster/scratch/nureasoning
export PYTHONPATH=$PWD/recon/devkit
OUT=${OUT:-runs_vla/q3vl2b_full_e5}; DATA=${DATA:-vla_full/train}
[ -d $OUT/final ] && { echo "already final"; exit 0; }
nvidia-smi --query-gpu=name --format=csv,noheader || exit 3
NG=$(nvidia-smi -L | wc -l)
LAST=$(ls -d $OUT/epoch_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
RES=""; [ -n "$LAST" ] && RES="--resume_from $LAST"
echo "resume: ${LAST:-none}  gpus: $NG"
env/bin/python -m torch.distributed.run --standalone --nproc_per_node=$NG -m nureasoning.nuvla.train   --data_root $DATA --test_data_root vla_data/validation --output_dir $OUT   --vlm_model_path Qwen/Qwen3-VL-2B-Instruct --epochs 5 --num_workers 3 --batch_size 1   --gradient_accumulation_steps 2 $RES
