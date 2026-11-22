"""1-GPU smoke test of Alpamayo 1.5 stage-2 fine-tuning on OUR data: load the trainable
model from the verified local config, build real samples with our adapter through the
recipe's own preprocessing + collate, one forward/backward, check only the action
expert gets gradients, report loss / memory / step time."""
import json, os, sys, time
import torch
from functools import partial
from hydra.utils import instantiate
from omegaconf import OmegaConf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a15ft_data import NuReasoningA15Dataset
from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1

REC = "/path/to/cluster/scratch/nureasoning/ext/alpamayo-recipes/recipes/alpamayo1_5_sft/configs"
model = TrainableAlpamayoR1.from_pretrained("models/a15_train_local", dtype=torch.bfloat16, cotrain_vlm=False).cuda()
print("vocab", model.config.vocab_size, "traj start", model.config.traj_token_start_idx)
ntr = sum(p.numel() for p in model.parameters() if p.requires_grad); ntot = sum(p.numel() for p in model.parameters())
print(f"trainable {ntr/1e6:.0f}M of {ntot/1e9:.2f}B")
pre = OmegaConf.load(f"{REC}/vla_processor/default.yaml"); pre["generation_mode"] = False
idx = [[c, 100] for c in sorted(os.listdir("img_a15/train"))[:4]]
json.dump(idx, open("artifacts/a15ft_index_smoke.json", "w"))
ds = NuReasoningA15Dataset("train", index_path="artifacts/a15ft_index_smoke.json", model_config=model.config,
                           vla_preprocess_args=OmegaConf.to_container(pre), img_roots=["img_a15", "img"])
from alpamayo.processor.qwen_processor import collate_fn_from_model_config
collate = partial(collate_fn_from_model_config, model_config=model.config, chat_template_version="r1_5")
batch = collate([ds[0], ds[1]])
def to(x):
    if torch.is_tensor(x): return x.cuda()
    if isinstance(x, dict): return {k: to(v) for k, v in x.items()}
    return x
batch = to(batch)
print("batch keys", sorted(batch.keys()))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], 1e-5)
for step in range(3):
    torch.cuda.synchronize(); t = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**{k: (dict(v) if isinstance(v, dict) else v) for k, v in batch.items()})
    out.loss.backward()
    g_exp = sum(float(p.grad.abs().sum()) for p in model.expert.parameters() if p.grad is not None)
    g_vlm = sum(1 for p in model.vlm.parameters() if p.grad is not None)
    opt.step(); opt.zero_grad(set_to_none=True); torch.cuda.synchronize()
    print(f"step {step}: loss {float(out.loss):.4f}  expert |grad| {g_exp:.3e}  vlm params with grad {g_vlm}  "
          f"{time.time()-t:.2f}s  peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
