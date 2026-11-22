#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J feat
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH --exclude=ins082
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/feat_%x_%j.out
source /etc/profile
export HF_HOME=/path/to/cluster/scratch/nureasoning/.cache/hf PYTHONNOUSERSITE=1
cd /path/to/cluster/scratch/nureasoning
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "BAD GPU NODE $(hostname)"; exit 3; }
/path/to/cluster/scratch/nureasoning/env/bin/python -c "import torch,sys; ok=torch.cuda.is_available(); print('cuda ok' if ok else 'CUDA UNAVAILABLE', torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 3)" || exit 3
/path/to/cluster/scratch/nureasoning/env/bin/python -u src/feat_extract.py --split $SPLIT --model $MODEL --bs 4 --workers 8
