#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J a15img
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 4:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/a15img_%j.out
source /etc/profile
export HF_HOME=/path/to/cluster/scratch/nureasoning/.cache/hf PYTHONNOUSERSITE=1
cd /path/to/cluster/scratch/nureasoning
env/bin/python -u src/img_ingest2.py --members artifacts/a15_members.json --splits validation --out img_a15 --save-index --clips 6 --K 24
