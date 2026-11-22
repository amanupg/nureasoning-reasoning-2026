#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J cpurun
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=96G
#SBATCH -t 4:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/cpurun_%j.out
source /etc/profile
export PYTHONNOUSERSITE=1
cd /path/to/cluster/scratch/nureasoning
env/bin/python -u "$@"
