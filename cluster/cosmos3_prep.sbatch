#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -c 16
#SBATCH --mem=100G
#SBATCH -t 8:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/cosmos3_prep_%j.out
# 2026-11-13: download Cosmos3-Nano / -Super weights (transformer/ + vision_encoder/ + root configs/tokenizer;
# OpenMDW-1.1, not gated) and extract the understanding tower to plain Qwen3-VL checkpoints (src/cosmos3_convert.py).
# usage: sbatch cluster/cosmos3_prep.sbatch Nano [Super]
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export HF_HOME=$P/.cache/hf PYTHONNOUSERSITE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DOWNLOAD_TIMEOUT=60
for r in "$@"; do
  S=$(env/bin/python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('nvidia/Cosmos3-$r', allow_patterns=['*.json', '*.jinja', '*.txt', 'transformer/*', 'vision_encoder/*', 'text_tokenizer/*'], max_workers=8))" | tail -1)
  echo "[dl] $r -> $S"; du -sh $S/ $S/transformer $S/vision_encoder 2>/dev/null
  low=$(echo $r | tr A-Z a-z)
  [ -f models/cosmos3_${low}_vlm/model.safetensors.index.json ] || env/bin/python -u src/cosmos3_convert.py $S models/cosmos3_${low}_vlm
  du -sh models/cosmos3_${low}_vlm
done
echo PREP_DONE
