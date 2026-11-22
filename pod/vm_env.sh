#!/bin/bash
# VM env: same package versions as Insomnia, torch built for CUDA >= 12.8 (B200 = sm_100)
set -x
sudo apt-get update -qq && sudo apt-get install -y -qq python3.12-venv pigz >/dev/null
python3 -m venv ~/env && . ~/env/bin/activate && pip install -q -U pip
for cu in cu130 cu129 cu128; do
  pip install -q torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/$cu && { echo "TORCH_FROM $cu"; break; }
done
pip install -q transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 safetensors==0.8.0 tokenizers==0.23.2 \
  pillow==12.3.0 numpy==2.5.2 flash-linear-attention==0.5.2 fla-core==0.5.2 huggingface_hub==1.33.0
pip list 2>/dev/null | grep -i -E "^(torch|torchvision|transformers|peft|triton|flash|fla-core|numpy|pillow|accelerate|safetensors|tokenizers) "
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); x=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16); print('matmul ok', float((x@x).float().mean()))"
python -c "import fla; print('fla ok', fla.__version__)"
echo ENV_DONE
