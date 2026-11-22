#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 6:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/pack_for_pod_%j.out
# 2026-11-16: pack training data into 4 GB tar parts for a rented pod (large files move ~100x faster than small ones).
# One tar stream per dir, in parallel; parts in packs/<name>.tar.part-aa.. + packs/<name>.files (member count) + MD5SUMS.
# usage: DIRS="img_vlm/train img_dense_A/train" sbatch cluster/pack_for_pod.sbatch   (paths relative to the repo root)
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P; mkdir -p packs
for d in $DIRS; do
  n=$(echo $d | tr '/' '_')
  ( t0=$(date +%s); find $d -type f | sort > packs/$n.files
    tar cf - -T packs/$n.files | split -b 4G -d -a 3 - packs/$n.tar.part-
    echo "[pack] $d: $(wc -l < packs/$n.files) files, $(du -cb packs/$n.tar.part-* | tail -1 | cut -f1) bytes, $(( $(date +%s) - t0 )) s" ) &
done
wait
( cd packs; md5sum $(for d in $DIRS; do echo $(echo $d | tr '/' '_').tar.part-*; done) >> MD5SUMS )
echo PACK_DONE
