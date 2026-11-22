#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 2:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/trainset_v2_check_%j.out
# 2026-11-16: TrainSet init on the v2 data (key + passes A/B/C), the pod run's exact env; draws 200 examples and checks
# every image path exists. CPU only.
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export DENSE=1 DENSE_PASSES=ABC DENSE_INDEX=artifacts/dense_index_train_v2.json DENSE_QPACK=artifacts/dense_q_train_v2.jsonl \
  DENSE_KS=1,12 DENSE_KW=.1,.9 IMG_ROOT=img_vlm PYTHONNOUSERSITE=1
env/bin/python -u - <<'PY'
import os, sys, collections; sys.path.insert(0, "src")
from vlm_data_dense import TrainSet, clip_images
ts = TrainSet("train", seed=0)
fr = collections.Counter(k[1] == 100 for k in ts.keys); print("keys", len(ts.keys), "key/extra", fr[True], fr[False], "clips", len({k[0] for k in ts.keys}))
miss = n = 0; typ = collections.Counter()
for i in range(200):
    (split, key), qs = ts.draw(); cams = clip_images(split, key)
    for _, a, b in cams:
        n += 2; miss += (not os.path.isfile(a)) + (not os.path.isfile(b))
    typ.update(q["subcategory"] for q in qs)
print("draws 200, images", n, "missing", miss); print("types", dict(typ.most_common()))
print("CHECK_OK" if miss == 0 and fr[False] > 100000 else "CHECK_FAIL")
PY
