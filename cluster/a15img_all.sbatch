#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J a15all
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/a15all_%j.out
source /etc/profile
export HF_HOME=/path/to/cluster/scratch/nureasoning/.cache/hf PYTHONNOUSERSITE=1
cd /path/to/cluster/scratch/nureasoning
srun --ntasks-per-node=1 bash -c "env/bin/python -u src/img_ingest2.py --members artifacts/a15_members_all.json --splits test,validation,train --out img_a15 --index-from img --save-index --clips 16 --K 16 > logs/a15all_node\${SLURM_PROCID}.log 2>&1"
