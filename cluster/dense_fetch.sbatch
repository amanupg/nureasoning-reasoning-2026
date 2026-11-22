#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -N 1
#SBATCH --array=0-7
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/dense_fetch_%A_%a.out
# 2026-11-14 dense-frame VLM run: images at the extra train frames (src/dense_frames.py), all 8 cameras at t and t-1 s,
# shrunk on arrival to 640x416 (the VLM key-frame size; t-1 s is resized further at load). One task per node (HF caps
# bandwidth per node; array tasks = shards, so each starts when a node frees). PASS=A: decision frames (~50, ~150) + missing key frames, saves tar indexes;
# PASS=B: the other three frames, reusing pass A's indexes. Resumable (_done.json per clip and pass).
# usage: PASS=A sbatch cluster/dense_fetch.sbatch ; PASS=B sbatch --dependency=afterany:<A> cluster/dense_fetch.sbatch
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export HF_HOME=$P/.cache/hf PYTHONNOUSERSITE=1 IMG_RESIZE=640x416
IDX="img_index/img:img_index/img_a15"; [ "$PASS" = B ] && IDX="img_dense_A:$IDX"; [ "$PASS" = C ] && IDX="img_dense_A:img_dense_B:$IDX"
env/bin/python -u src/img_ingest2.py --members artifacts/dense_members_train_$PASS.json --splits train --out img_dense_$PASS --index-from $IDX --save-index --clips 16 --K 16 --shard $SLURM_ARRAY_TASK_ID --nshards ${NSHARDS:-8}
echo DENSE_FETCH_${PASS}_DONE
