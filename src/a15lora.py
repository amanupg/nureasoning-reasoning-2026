"""LoRA for Alpamayo 1.5 stage 1 (the recipe only has full VLM fine-tuning).

train:  model._target_ = a15lora.lora_from_alpamayo_checkpoint (Hydra), same args as
        TrainableReasoningVLA.from_alpamayo_checkpoint plus lora_r / lora_alpha / lora_dropout.
        Adapters go on the LANGUAGE model's attention and MLP projections only; the vision
        encoder, embeddings (incl. trajectory tokens) and lm_head stay frozen.
merge:  python src/a15lora.py merge <stage1_checkpoint_dir> <out_dir> <r> <alpha>
        folds W += (alpha / r) * B @ A into the base weights and writes a plain checkpoint with
        the original key names, so the recipe's stage 2 (model.stage1_vlm_checkpoint_path) and
        our inference load it unchanged.
"""
import json, os, re, sys

TARGET = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def lora_from_alpamayo_checkpoint(checkpoint_path, vlm_name_or_path, lora_r=32, lora_alpha=64,
                                  lora_dropout=0.05, **kwargs):
    from peft import LoraConfig, inject_adapter_in_model
    from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA
    m = TrainableReasoningVLA.from_alpamayo_checkpoint(checkpoint_path, vlm_name_or_path, **kwargs)
    for p in m.parameters():
        p.requires_grad = False
    cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=TARGET, bias="none")
    m.vlm = inject_adapter_in_model(cfg, m.vlm)          # module names kept; adds lora_A / lora_B
    if hasattr(m.vlm, "enable_input_require_grads"):
        m.vlm.enable_input_require_grads()               # gradient checkpointing with a frozen base
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in m.parameters())
    print(f"[a15lora] trainable {n_tr:,} / {n_all:,} ({100 * n_tr / n_all:.2f}%), r={lora_r} alpha={lora_alpha}", flush=True)
    m._a15lora = {"r": lora_r, "alpha": lora_alpha, "target": TARGET}
    return m


def merge(src, out, r, alpha):
    """Fold LoRA into base weights; write safetensors shards with the original key names."""
    import torch
    from safetensors.torch import load_file, save_file
    idx = os.path.join(src, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.isfile(idx) else ["model.safetensors"]
    sd = {}
    for f in files:
        sd.update(load_file(os.path.join(src, f)))
    scale = float(alpha) / float(r)
    assert any(".lora_A." in k for k in sd), "no LoRA weights in this checkpoint"
    out_sd, n = {}, 0
    for k, v in sd.items():
        if ".lora_A." in k or ".lora_B." in k:
            continue
        if k.endswith(".base_layer.weight"):
            stem = k[: -len(".base_layer.weight")]
            A = sd[f"{stem}.lora_A.default.weight"].float(); B = sd[f"{stem}.lora_B.default.weight"].float()
            out_sd[stem + ".weight"] = (v.float() + scale * (B @ A)).to(v.dtype); n += 1
        elif k.endswith(".base_layer.bias"):
            out_sd[k.replace(".base_layer.bias", ".bias")] = v
        else:
            out_sd[k] = v
    os.makedirs(out, exist_ok=True)
    keys = sorted(out_sd); shard, size, shards, wm = {}, 0, [], {}
    for k in keys:
        shard[k] = out_sd[k].contiguous(); size += out_sd[k].numel() * out_sd[k].element_size()
        if size > 4.5e9:
            shards.append(shard); shard, size = {}, 0
    if shard: shards.append(shard)
    for i, s in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(s, os.path.join(out, name), metadata={"format": "pt"})
        wm.update({k: name for k in s})
    json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
    for f in os.listdir(src):
        if f.endswith(".json") and f not in ("model.safetensors.index.json", "trainer_state.json"):
            os.system(f"cp {os.path.join(src, f)} {out}/")
    print(f"merged {n} LoRA layers (scale {scale}) -> {out} ({len(shards)} shards)")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "merge":
    merge(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
