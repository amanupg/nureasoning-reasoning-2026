#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J img2
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/img2_%j.out
source /etc/profile
export HF_HOME=/path/to/cluster/scratch/nureasoning/.cache/hf PYTHONNOUSERSITE=1
cd /path/to/cluster/scratch/nureasoning
srun --ntasks-per-node=1 bash -c "/path/to/cluster/scratch/nureasoning/env/bin/python -u src/img_ingest2.py --clips 4 --K 24 --save-index > /path/to/cluster/scratch/nureasoning/logs/img2_node\${SLURM_PROCID}.log 2>&1"
