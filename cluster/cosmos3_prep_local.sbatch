#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -c 16
#SBATCH --mem=100G
#SBATCH -t 8:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/cosmos3_prep_local_%j.out
# Same as cosmos3_prep.sbatch, but downloads to node-local /local (GPFS writes ran ~5 MB/s) and writes only the
# extracted understanding tower to models/. usage: sbatch cluster/cosmos3_prep_local.sbatch Nano [Super]
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
L=/local/$USER_${SLURM_JOB_ID}; mkdir -p $L; trap "rm -rf $L" EXIT
export HF_HOME=$L/hf PYTHONNOUSERSITE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DOWNLOAD_TIMEOUT=60
df -h /local | tail -1
for r in "$@"; do
  low=$(echo $r | tr A-Z a-z)
  [ -f models/cosmos3_${low}_vlm/model.safetensors.index.json ] && { echo "[skip] $r done"; continue; }
  t0=$(date +%s)
  S=$(env/bin/python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('nvidia/Cosmos3-$r', allow_patterns=['*.json', '*.jinja', '*.txt', 'transformer/*', 'vision_encoder/*', 'text_tokenizer/*'], max_workers=16))" | tail -1)
  echo "[dl] $r -> $S in $(( $(date +%s) - t0 )) s, $(du -sh $S/ | cut -f1)"
  env/bin/python -u src/cosmos3_convert.py $S $L/cosmos3_${low}_vlm && cp -r $L/cosmos3_${low}_vlm models/cosmos3_${low}_vlm.tmp && mv models/cosmos3_${low}_vlm.tmp models/cosmos3_${low}_vlm
  du -sh models/cosmos3_${low}_vlm; rm -rf $S $L/hf $L/cosmos3_${low}_vlm
done
echo PREP_DONE
