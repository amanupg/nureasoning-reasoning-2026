#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J crop90
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/crop90_%j.out
# Full-resolution crops around box3d samples (artifacts/box3d_crops.json); full frames deleted after cropping.
source /etc/profile
export HF_HOME=/path/to/cluster/scratch/nureasoning/.cache/hf PYTHONNOUSERSITE=1 IMG_CROPS=artifacts/box3d_crops_f90.json
cd /path/to/cluster/scratch/nureasoning
srun --ntasks-per-node=1 bash -c "env/bin/python -u src/img_ingest2.py --members src/box3d_crop_members_f90.json --splits test,validation,train --out img_crops90 --index-from img:img_a15 --clips 16 --K 16 > logs/crop90_node\${SLURM_PROCID}.log 2>&1"
