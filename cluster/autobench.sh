#!/bin/bash
# Submit a validation benchmark for every new nuVLA epoch checkpoint of the given runs.
cd /path/to/cluster/scratch/nureasoning; source /etc/profile
while true; do
  for h in h0 h1; do for e in 1 2 3 4 5; do
    d=runs_vla/q3vl2b_full_$h/epoch_$e; b=runs_vla/bench_full_${h}_e$e
    if [ -f $d/action_expert.pt ] && [ ! -e $b.submitted ]; then
      sleep 60; sbatch -J bench_${h}e$e -t 2:00:00 src/gpu_run.sbatch -m nureasoning.planning.benchmark --data_root vla_data/validation --mode vla --checkpoint_dir $d --output_dir $b --vis_clips 0 && touch $b.submitted
    fi
  done; done
  squeue -u "$USER" -h -o "%j" | grep -q "^nv_" || break
  sleep 600
done
