#!/bin/bash
# 2026-11-16 rented-pod setup for the dense continuation (8x H100 SXM, RunPod). Run from /workspace/nureasoning.
# 1) exact env of the Insomnia runs; 2) unpack data parts received from Insomnia (packs2/ chunks + packs/ model),
# verify md5 + member counts per chunk.
set -e
pip install -q torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
pip install -q transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 safetensors==0.8.0 tokenizers==0.23.2 \
  pillow==12.3.0 numpy==2.5.2 flash-linear-attention==0.5.2 fla-core==0.5.2 triton==3.8.0 huggingface_hub==1.33.0
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))"
for D in packs packs2; do ( cd $D && md5sum -c MD5SUMS --quiet && echo "$D MD5_OK" ); done
for f in packs/models_qwen_drive_vlm.files packs2/*.c*.files; do n=${f%.files}
  ls $n.tar.part-* >/dev/null 2>&1 || continue
  cat $n.tar.part-* | tar xf -
  python - "$f" <<'PY'
import os, sys; L = [l.rstrip("\n") for l in open(sys.argv[1])]; got = sum(os.path.isfile(p) for p in L)
print(f"[unpack] {sys.argv[1]}: {got} / {len(L)} files"); sys.exit(got != len(L))
PY
done
tar xf code_artifacts.tar      # src/, pod/, artifacts needed by TrainSet, init_adapter/
echo SETUP_DONE
