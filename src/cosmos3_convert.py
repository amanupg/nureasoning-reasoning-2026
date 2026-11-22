"""Extract the UNDERSTANDING tower of NVIDIA Cosmos3 (Nano / Super; Mixture-of-Transformers omni model) into a plain
transformers Qwen3VLForConditionalGeneration checkpoint, so our LoRA SFT / inference code (src/vlm_sft.py,
src/vlm_infer.py, AutoModelForImageTextToText) runs on it unchanged.

Mapping (diffusers transformer_cosmos3.py: the causal "und" pathway uses to_q/to_k/to_v/to_out + norm_q/norm_k,
mlp, input_layernorm, post_attention_layernorm; the *_moe_gen / add_* / norm_added_* weights are the generation
tower and are dropped; text hidden_act silu -> plain RMSNorm, identical to Qwen3):
  transformer/ embed_tokens, norm, lm_head, layers.N.{input_layernorm, post_attention_layernorm, mlp.*}
      -> model.language_model.<same>  (lm_head -> lm_head)
  transformer/ layers.N.self_attn.{to_q,to_k,to_v,to_out,norm_q,norm_k}
      -> model.language_model.layers.N.self_attn.{q_proj,k_proj,v_proj,o_proj,q_norm,k_norm}
  vision_encoder/ * -> model.visual.*
Config: Qwen3VLConfig from the Cosmos3 root config's text_config / vision_config / token ids.
usage: python src/cosmos3_convert.py <Cosmos3 snapshot dir> <out dir>
"""
import glob, json, os, re, shutil, sys
from safetensors import safe_open
from safetensors.torch import save_file

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
root = json.load(open(os.path.join(src, "config.json")))
tc, vc = dict(root["text_config"]), dict(root["vision_config"])
assert tc.get("hidden_act", "silu") == "silu", tc.get("hidden_act")       # relu2 would mean Nemotron norms
tc["model_type"] = "qwen3_vl_text"; vc["model_type"] = "qwen3_vl"
cfg = {"architectures": ["Qwen3VLForConditionalGeneration"], "model_type": "qwen3_vl", "text_config": tc,
       "vision_config": vc, "tie_word_embeddings": root.get("tie_word_embeddings", False)}
for k in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
    cfg[k] = root[k]
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)

ATT = {"to_q": "q_proj", "to_k": "k_proj", "to_v": "v_proj", "to_out": "o_proj", "norm_q": "q_norm", "norm_k": "k_norm"}


def map_text(k):
    if k in ("embed_tokens.weight", "norm.weight"): return "model.language_model." + k
    if k == "lm_head.weight": return k
    m = re.match(r"layers\.(\d+)\.(.+)$", k)
    if not m: return None
    i, rest = m.groups()
    if "moe_gen" in rest: return None
    if rest.split(".")[0] in ("input_layernorm", "post_attention_layernorm", "mlp"):
        return f"model.language_model.layers.{i}.{rest}"
    m2 = re.match(r"self_attn\.(to_q|to_k|to_v|to_out|norm_q|norm_k)\.(weight|bias)$", rest)
    if m2: return f"model.language_model.layers.{i}.self_attn.{ATT[m2.group(1)]}.{m2.group(2)}"
    return None


sd, dropped = {}, 0
for f in sorted(glob.glob(os.path.join(src, "transformer", "*.safetensors"))):
    with safe_open(f, "pt") as h:
        for k in h.keys():
            nk = map_text(k)
            if nk is None: dropped += 1; continue
            sd[nk] = h.get_tensor(k).contiguous()
with safe_open(os.path.join(src, "vision_encoder", "model.safetensors"), "pt") as h:
    for k in h.keys(): sd["model.visual." + k] = h.get_tensor(k).contiguous()
L = tc["num_hidden_layers"]
need = [f"model.language_model.layers.{i}.self_attn.{p}.weight" for i in range(L) for p in ATT.values()]
missing = [k for k in need if k not in sd]
assert not missing, missing[:5]
print(f"kept {len(sd)} tensors ({sum(v.numel() for v in sd.values()) / 1e9:.2f} B params), dropped {dropped}", flush=True)
# shard ~5 GB
shards, cur, size = [], {}, 0
for k in sorted(sd):
    n = sd[k].numel() * sd[k].element_size()
    if cur and size + n > 5e9: shards.append(cur); cur, size = {}, 0
    cur[k] = sd[k]; size += n
shards.append(cur)
wm = {}
for j, s in enumerate(shards):
    name = f"model-{j + 1:05d}-of-{len(shards):05d}.safetensors"
    save_file(s, os.path.join(out, name), metadata={"format": "pt"}); wm.update({k: name for k in s})
json.dump({"metadata": {"total_size": sum(v.numel() * v.element_size() for v in sd.values())}, "weight_map": wm},
          open(os.path.join(out, "model.safetensors.index.json"), "w"))
for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja", "chat_template.json",
          "preprocessor_config.json", "video_preprocessor_config.json", "generation_config.json", "special_tokens_map.json",
          "added_tokens.json"):
    if os.path.isfile(os.path.join(src, f)): shutil.copy(os.path.join(src, f), os.path.join(out, f))
print("->", out, sorted(os.listdir(out)), flush=True)
