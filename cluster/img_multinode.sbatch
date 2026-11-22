#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J img_all
#SBATCH -N 20
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 08:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/img_all_%j.out
# One task per node: the HF bandwidth cap is ~112 MB/s PER NODE (measured), and
# 4 streams saturate a node, so throughput scales with distinct nodes.
source /etc/profile
export HF_HOME=/path/to/cluster/scratch/nureasoning/.cache/hf PYTHONNOUSERSITE=1
cd /path/to/cluster/scratch/nureasoning
srun --ntasks-per-node=1 bash -c "for S in test validation train; do /path/to/cluster/scratch/nureasoning/env/bin/python src/img_ingest.py --members src/image_members.json --split \$S --out img --shard \$SLURM_PROCID --nshards \$SLURM_NTASKS --workers 4 > /path/to/cluster/scratch/nureasoning/logs/img_\${S}_node\${SLURM_PROCID}.log 2>&1; done"
