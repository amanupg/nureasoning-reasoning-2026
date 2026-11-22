#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH --requeue
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/pack_for_pod2_%j.out
# 2026-11-16: like pack_for_pod.sbatch but each dir is cut into NCH file-list chunks, one tar stream per chunk
# (GPFS small-file reads are latency-bound, so more streams = more throughput). Parts: packs2/<name>.cK.tar.part-NNN,
# packs2/<name>.cK.files (member list); packs2/MD5SUMS written at the end. Rerun-safe: a chunk with a .ok marker is skipped.
# usage: DIRS="img_dense_C/train img_vlm/train" NCH=6 sbatch cluster/pack_for_pod2.sbatch
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P; mkdir -p packs2; NCH=${NCH:-6}
for d in $DIRS; do
  n=$(echo $d | tr '/' '_')
  [ -f packs2/$n.list ] || { find $d -type f | sort > packs2/$n.list.tmp && mv packs2/$n.list.tmp packs2/$n.list; }
  L=$(wc -l < packs2/$n.list); per=$(( (L + NCH - 1) / NCH ))
  for k in $(seq 0 $((NCH - 1))); do
    [ -f packs2/$n.c$k.ok ] && continue
    ( sed -n "$((k * per + 1)),$(((k + 1) * per))p" packs2/$n.list > packs2/$n.c$k.files
      rm -f packs2/$n.c$k.tar.part-*; t0=$(date +%s)
      tar cf - -T packs2/$n.c$k.files | split -b 4G -d -a 3 - packs2/$n.c$k.tar.part- && touch packs2/$n.c$k.ok
      echo "[pack] $d c$k: $(wc -l < packs2/$n.c$k.files) files, $(du -cb packs2/$n.c$k.tar.part-* | tail -1 | cut -f1) bytes, $(( $(date +%s) - t0 )) s" ) &
  done
done
wait
( cd packs2; ls *.tar.part-* | sort | xargs md5sum > MD5SUMS )
echo PACK_DONE
