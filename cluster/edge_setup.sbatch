#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J edge_setup
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 3:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/edge_setup_%j.out
# Alpamayo2-Edge zero-shot check (2026-11-11), CPU part: uv env (torch 2.8, transformers 5.17, as the model card pins),
# code at a fixed commit, weights at the README's pinned revision (not gated; 5.3 GB), env tar for fast staging,
# member list and resized frames (544x352, inside the VQA pixel budget) into img_edge/. Idempotent.
source /etc/profile; set -x
P=/path/to/cluster/scratch/nureasoning; cd $P
export UV_CACHE_DIR=$P/.cache/uv HF_HOME=$P/.cache/hf UV_PYTHON_INSTALL_DIR=$P/.cache/uvpy PYTHONNOUSERSITE=1
if [ ! -f ext/edge_env.ok ]; then
  [ -d ext/alpamayo2-edge ] || git clone -q https://github.com/NVlabs/alpamayo2-edge ext/alpamayo2-edge
  git -C ext/alpamayo2-edge checkout -q 02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e || exit 2
  ~/.local/bin/uv venv -p 3.12 ext/edge_env
  ~/.local/bin/uv pip install -p ext/edge_env/bin/python "torch==2.8.0" "torchvision==0.23.0" "transformers==5.17.0" \
      "accelerate>=1.12.0" "huggingface-hub>=1.0,<2" "safetensors>=0.8.0" "einops>=0.8.1" "hydra-core>=1.3.2" \
      "numpy>=2.0" "pillow>=11.0" "scipy>=1.16.0" "colorlog>=6.0" || exit 3
  ~/.local/bin/uv pip install -p ext/edge_env/bin/python --no-deps -e ext/alpamayo2-edge || exit 4
  ext/edge_env/bin/python -c "import torch, transformers, alpamayo2_edge; print('[setup] ok', torch.__version__, transformers.__version__)" || exit 5
  touch ext/edge_env.ok
fi
if [ ! -f models/alpamayo2_edge/config.json ]; then
  ext/edge_env/bin/python - <<'PY' || exit 6
from huggingface_hub import snapshot_download
snapshot_download(repo_id="nvidia/Alpamayo2-Edge", revision="f8951d095de3589b3face1dec3f4394c8ecedb38",
                  local_dir="models/alpamayo2_edge")
print("[setup] weights ok")
PY
fi
[ -f ext/edge_env.tar ] && [ ext/edge_env.tar -nt ext/edge_env.ok ] || tar -C ext -cf ext/edge_env.tar edge_env
env/bin/python src/edge_members.py --val-n ${VAL_N:-250} ${EDGE_TEST:+--test} || exit 7
IMG_RESIZE=544x352 env/bin/python -u src/img_ingest2.py --members src/edge_members.json --splits validation${EDGE_TEST:+,test} \
    --out img_edge --index-from img_index/img:img_index/img_a15:img_index/img_full --clips 8 --K 24 2>&1 | grep -E "DONE|FAIL|^\[img2\]|\]  ok" | tail -20
du -sh img_edge models/alpamayo2_edge ext/edge_env.tar
echo SETUP_DONE
