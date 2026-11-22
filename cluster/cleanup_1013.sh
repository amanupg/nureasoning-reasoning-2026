#!/bin/bash
# Approved by Aman 2026-10-13: items 1-4 (img_a15ft; runs_a15ft/h0,h1; runs_a15s1/lora/checkpoint-1740;
# img_dist, VLM smoke dirs, intermediate VLM checkpoints). Inventory first, verify kept items, then delete.
set -e
N=/path/to/cluster/scratch/nureasoning; cd $N; L=logs/cleanup_2026-10-13.txt
for k in runs_a15ft/all/checkpoint-30573 runs_a15s1/lora/merged runs_vlm/qd4b_r64/step_3000 runs_vlm/qd4b_tv/step_2750; do
  test -d $k && echo "kept-present $k ($(ls $k | wc -l) entries)" || { echo "MISSING kept item $k - abort"; exit 1; }
done
DEL="img_a15ft runs_a15ft/h0 runs_a15ft/h1 runs_a15s1/lora/checkpoint-1740 img_dist runs_vlm/smoke_sft runs_vlm/smoke_pf"
DEL="$DEL $(ls -d runs_vlm/smoke_zs_* 2>/dev/null)"
for s in 250 500 750 1000 1250 1500 1750 2000 2250 2500 2750; do DEL="$DEL runs_vlm/qd4b_r64/step_$s"; done
for s in 250 500 750 1000 1250 1500 1750 2000 2250 2500; do DEL="$DEL runs_vlm/qd4b_tv/step_$s"; done
: > $L
for d in $DEL; do [ -e $d ] && find $d -type f >> $L; done
echo "inventory: $(wc -l < $L) files -> $L"
for d in $DEL; do [ -e $d ] && rm -rf $d; done
echo "deleted. remaining: $(ls runs_a15ft) | $(ls runs_a15s1/lora) | $(ls runs_vlm/qd4b_r64) | $(ls runs_vlm/qd4b_tv)"
df -h /path/to/cluster | tail -1
