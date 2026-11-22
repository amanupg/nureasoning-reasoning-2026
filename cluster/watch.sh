#!/bin/bash
# CPU watcher: submits GPU work only when its inputs exist, so no GPU sits idle.
source /etc/profile; cd /path/to/cluster/scratch/nureasoning
n() { ls img/$1/*/_done.json 2>/dev/null | wc -l; }
until [ $(n test) -ge 995 ]; do sleep 60; done
echo "$(date -u) test images ready: $(n test)"
sbatch --parsable -J feat_test --export=ALL,SPLIT=test,MODEL=facebook/dinov3-vitb16-pretrain-lvd1689m src/feat.sbatch
until [ $(n validation) -ge 995 ] || [ $(n train) -ge 20 ]; do sleep 60; done
echo "$(date -u) validation images ready: $(n validation)"
F=$(sbatch --parsable -J feat_val --export=ALL,SPLIT=validation,MODEL=facebook/dinov3-vitb16-pretrain-lvd1689m src/feat.sbatch)
echo "feat_val job $F"
for NV in "" "--no-vision"; do
  sbatch --parsable -d afterok:$F -J vs_dbg --export=ALL,ARGS="--train validation --eval validation --train-frac 0.7 --epochs 40 $NV --out runs_v/debug" src/vscorer.sbatch
done
echo "$(date -u) watcher done"
