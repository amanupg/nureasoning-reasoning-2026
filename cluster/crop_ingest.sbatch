#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J crop_ingest
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/crop_ingest_%j.out
# Full-resolution crops around box3d samples (artifacts/box3d_crops.json); full frames deleted after cropping.
source /etc/profile
export HF_HOME=/path/to/cluster/scratch/nureasoning/.cache/hf PYTHONNOUSERSITE=1 IMG_CROPS=artifacts/box3d_crops.json
cd /path/to/cluster/scratch/nureasoning
srun --ntasks-per-node=1 bash -c "env/bin/python -u src/img_ingest2.py --members src/box3d_crop_members.json --splits test,validation,train --out img_crops --index-from img:img_a15 --clips 16 --K 16 > logs/crop_ingest_node\${SLURM_PROCID}.log 2>&1"
