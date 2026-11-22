#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=ins082,ins084,ins046,ins047
#SBATCH -t 2:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/vlm_smoke_%j.out
# Smoke: convert Qwen-Drive VLM, zero-shot inference on 40 val questions, 20 LoRA steps (speed / memory).
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
export HF_HOME=$P/.cache/hf HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH=$P/recon/devkit
export TMPDIR=/dev/shm/$USER_${SLURM_JOB_ID}; mkdir -p $TMPDIR; trap "rm -rf $TMPDIR" EXIT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=$P/models/qwen_drive_vlm
if [ ! -f $M/model.safetensors ]; then
  env/bin/python src/vlm_convert.py $(ls -d .cache/hf/hub/models--Qwen--Qwen-Drive-1.0-4B/snapshots/*) $M
fi
mkdir -p runs_vlm
[ -f runs_vlm/smoke_zs_s0of1.jsonl ] || env/bin/python -u src/vlm_infer.py validation $M runs_vlm/smoke_zs --limit 40 2>&1 | grep -v -i warn
env/bin/python - <<'PY' 2>&1 | grep -v -i -E "warn|processor_kwargs"
import sys; sys.path.insert(0, "src")
from transformers import AutoProcessor
from vlm_data import Encoder, load_questions, clip_images
enc = Encoder(AutoProcessor.from_pretrained("models/qwen_drive_vlm"))
Q = load_questions("validation"); c = sorted(Q)[0]
b, pos = enc(clip_images("validation", c), Q[c][:2], [Q[c][0]["answer"], Q[c][1]["answer"]])
t = enc.p.tokenizer; ids = b["input_ids"][0]
print("keys", list(b.keys()), "len", ids.shape[0], "image tokens", int((ids == enc.p.tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum()))
print("TAIL:", repr(t.decode(ids[-40:])))
print("label positions", [(p + 1, t.decode([int(ids[p + 1])])) for p in pos])
PY
env/bin/python -u src/vlm_sft.py --model $M --out runs_vlm/smoke_sft --steps 20 --accum 2 --save_every 20 2>&1 | grep -v -i warn
nvidia-smi --query-gpu=name,memory.total --format=csv
