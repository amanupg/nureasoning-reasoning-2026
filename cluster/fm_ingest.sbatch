#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J fm_ingest
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/fm_ingest_%j.out
# Frames 90..100 of the cameras named by future_motion_label questions (src/fm_members.json),
# validation + test, half resolution, into img_fm/. Reuses saved tar indexes (img, img_a15).
source /etc/profile
export HF_HOME=/path/to/cluster/scratch/nureasoning/.cache/hf PYTHONNOUSERSITE=1 IMG_RESIZE=1408x928
cd /path/to/cluster/scratch/nureasoning
srun --ntasks-per-node=1 bash -c "env/bin/python -u src/img_ingest2.py --members src/fm_members.json --splits test,validation --out img_fm --index-from img:img_a15 --clips 16 --K 16 > logs/fm_ingest_node\${SLURM_PROCID}.log 2>&1"
