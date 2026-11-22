#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J vlafull
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/vlafull_%j.out
source /etc/profile
export HF_HOME=/path/to/cluster/scratch/nureasoning/.cache/hf PYTHONNOUSERSITE=1 IMG_RESIZE=896x592
cd /path/to/cluster/scratch/nureasoning
srun --ntasks-per-node=1 bash -c "env/bin/python -u src/img_ingest2.py --members artifacts/vla_members_train.json --splits train --out img_full --index-from img:img_a15 --clips 16 --K 16 > logs/vlafull_node\${SLURM_PROCID}.log 2>&1"
