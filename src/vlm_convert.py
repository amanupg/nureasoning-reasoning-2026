"""Extract the plain Qwen3.5 VLM from a Qwen-Drive-1.0 checkpoint (QwenDriveForPlanning wraps it as
`vlm.*`): writes config.json (= vlm_config), model.safetensors (prefix stripped) and the tokenizer /
processor files into <out>, loadable with transformers Qwen3_5ForConditionalGeneration.
usage: python src/vlm_convert.py <qwen-drive snapshot dir> <out dir>
"""
import json, os, shutil, sys
from safetensors.torch import load_file, save_file

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
cfg = json.load(open(os.path.join(src, "config.json")))
json.dump(cfg["vlm_config"], open(os.path.join(out, "config.json"), "w"), indent=2)
sd = load_file(os.path.join(src, "model.safetensors"))
vlm = {k[len("vlm."):]: v for k, v in sd.items() if k.startswith("vlm.")}
print(f"{len(sd)} tensors, {len(vlm)} under vlm.")
save_file(vlm, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
          "chat_template.json", "preprocessor_config.json", "generation_config.json"):
    if os.path.isfile(os.path.join(src, f)):
        shutil.copy(os.path.join(src, f), os.path.join(out, f))
print("->", out, sorted(os.listdir(out)))
