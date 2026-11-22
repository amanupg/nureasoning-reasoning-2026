#!/bin/bash
# Merge the latest LoRA stage-1 checkpoint of <run_dir> into <run_dir>/merged (plain checkpoint for stage 2).
cd /path/to/cluster/scratch/nureasoning
R=$1; CK=$(ls -d $R/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
[ -z "$CK" ] && { echo "no checkpoint in $R"; exit 2; }
echo "merging $CK"; rm -rf $R/merged
env/bin/python src/a15lora.py merge $CK $R/merged ${2:-32} ${3:-64}
