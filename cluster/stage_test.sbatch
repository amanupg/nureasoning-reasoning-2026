#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ins058,ins082,ins084,ins046,ins047
#SBATCH -t 3:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/stage_test_%j.out
# Phase-1 check: stage the test frames to node-local disk, verify they are byte-identical to img/test, then
# rerun VLM shard 0/4 (run 2 adapter) from the staged frames and compare with the stored predictions.
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export HF_HOME=$P/.cache/hf PYTHONNOUSERSITE=1 PYTHONPATH=$P/recon/devkit
source cluster/stage_img.sh test
env/bin/python - <<'PY'
import hashlib, os, glob
root = os.environ["IMG_ROOT"]; n = same = miss = 0
for p in glob.glob("img/test/*/cameras/*/*.jpg"):
    q = os.path.join(root, os.path.relpath(p, "img")); n += 1
    if not os.path.isfile(q): miss += 1; continue
    same += hashlib.sha256(open(p, "rb").read()).digest() == hashlib.sha256(open(q, "rb").read()).digest()
print(f"[check] frames in img/test {n}: staged identical {same}, missing {miss}, different {n - same - miss}", flush=True)
PY
export HF_HUB_OFFLINE=1
env/bin/python -u src/vlm_infer.py test $P/models/qwen_drive_vlm runs_vlm/stagecheck_test --adapter $P/runs_vlm/qd4b_tv/step_2750 --shard 0 --nshards 4 2>&1 | grep --line-buffered -v -i -E "warn|processor_kwargs|regex" | tail -3
env/bin/python - <<'PY'
import json
A = {json.loads(l)["question_id"]: json.loads(l) for l in open("runs_vlm/qd4b_tv2750_test_s0of4.jsonl")}
B = {json.loads(l)["question_id"]: json.loads(l) for l in open("runs_vlm/stagecheck_test_s0of4.jsonl")}
k = set(A) & set(B); same = sum(A[q]["pred"] == B[q]["pred"] for q in k)
d = max(abs(A[q]["probs"][c] - B[q]["probs"][c]) for q in k for c in A[q]["probs"])
print(f"[check] VLM shard 0: {len(B)} vs {len(A)} questions, same answer {same}/{len(k)}, max prob diff {d:.2e}", flush=True)
PY
