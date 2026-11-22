#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 4:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/persist_img_%j.out
# 2026-11-13: keep one shared copy of the VLM frames (train,validation,test; 167 GB) so later VLM jobs skip the
# ~6 h throttled HF re-stage. Copies a running job's node-local stage. usage: sbatch -w <node> persist_img.sbatch <src dir>
P=/path/to/cluster/scratch/nureasoning; D=$P/img_vlm
mkdir -p $D.tmp; t0=$(date +%s)
rsync -a "$1"/ $D.tmp/ || exit 1
mv $D.tmp $D; echo "PERSIST_DONE $(du -sh $D | cut -f1) $(find $D -type f | wc -l) files in $(( $(date +%s) - t0 )) s"
