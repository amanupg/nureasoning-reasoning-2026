"""LoRA SFT of a Qwen3.5-family VLM on generated multiple-choice questions (src/vlm_data.py format).
torchrun --nproc_per_node N src/vlm_sft.py --model <dir> --out <dir> [--steps 3000 --accum 8 --lr 1e-4 --rank 64]
- LoRA on every Linear of the language model (attention, Gated DeltaNet, MLP); image encoder frozen unless --vision lora (env VISION)
- options are re-lettered at random per draw (no answer-position prior); loss on answer letters only
- resumable: adapter + optimizer + step in <out>/last/, snapshots in <out>/step_<n>/
"""
import argparse, json, math, os, random, time
import torch, torch.distributed as dist
HERE = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, HERE)
from vlm_data import Encoder, TrainSet, clip_images
if os.environ.get("DENSE") == "1":                  # 2026-11-14 dense-frame run: (clip, frame) examples
    from vlm_data_dense import TrainSet, clip_images


def shuffle_choices(q, rng):
    letters = sorted(q["choices"]); texts = [q["choices"][L] for L in letters]
    order = list(range(len(texts))); rng.shuffle(order)
    new = {letters[i]: texts[j] for i, j in enumerate(order)}
    ans = letters[order.index(letters.index(q["answer"]))]
    return dict(q, choices=new, answer=ans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=100); ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--state_every", type=int, default=0, help="also write <out>/last/state.pt every n steps (preemptible partitions)")
    ap.add_argument("--split", default="train"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init_adapter", default="", help="warm start from a saved adapter dir (ignored when resuming)")
    ap.add_argument("--vision", default=os.environ.get("VISION", "frozen"), choices=["frozen", "lora"],
                    help="lora: LoRA also on the image encoder's Linear layers (default frozen = earlier runs)")
    a = ap.parse_args()
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world = 0, 1
    gpr = int(os.environ.get("GPUS_PER_RANK", "1"))     # >1: each rank holds one layer-split copy over gpr GPUs
    first = int(os.environ.get("LOCAL_RANK", 0)) * gpr
    torch.cuda.set_device(first); dev = torch.device("cuda", first)
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, get_peft_model_state_dict
    proc = AutoProcessor.from_pretrained(a.model); enc = Encoder(proc)
    if gpr > 1:
        mm = {g: os.environ.get("GPU_MEM", "38GiB") for g in range(first, first + gpr)}
        model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="balanced", max_memory=mm)
    else:
        model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16).to(dev)
    model.config.use_cache = False
    names = sorted({n.split(".")[-1] for n, m in model.named_modules()
                    if isinstance(m, torch.nn.Linear) and "language_model" in n})
    pat = rf".*language_model.*\.({'|'.join(names)})"
    if a.vision == "lora":                               # 2026-11-14: also adapt the image encoder + merger
        vn = sorted({n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and ".visual." in f".{n}."})
        assert vn, "no Linear layers under 'visual'"
        pat = rf"({pat}|.*\.?visual\.(?!patch_embed).*\.({'|'.join(vn)}))"
        names = names + [f"visual:{v}" for v in vn]
    lcfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias="none", target_modules=pat + "$")
    model = get_peft_model(model, lcfg)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if rank == 0: model.print_trainable_parameters(); print("LoRA targets:", names, flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))
    step = 0; last = os.path.join(a.out, "last")
    if a.init_adapter and not os.path.isfile(os.path.join(last, "state.pt")):
        from safetensors.torch import load_file
        sd = load_file(os.path.join(a.init_adapter, "adapter_model.safetensors"))
        res = set_peft_model_state_dict(model, sd)
        if rank == 0: print(f"warm start from {a.init_adapter}: {len(sd)} tensors, unexpected {len(getattr(res, 'unexpected_keys', []))}", flush=True)
    if os.path.isfile(os.path.join(last, "state.pt")):
        st = torch.load(os.path.join(last, "state.pt"), map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, st["adapter"]); opt.load_state_dict(st["opt"]); step = st["step"]
        if rank == 0: print(f"resumed at step {step}", flush=True)
    data = TrainSet(a.split, seed=a.seed * 1000 + rank + 7919 * step); rng = random.Random(a.seed * 31 + rank + step)
    # CPU-side example preparation (JPEG decode, resize, processor) runs in background threads so the GPU
    # does not wait on it; each thread owns its rng. Order of examples is irrelevant (random draws).
    import queue, threading
    qbuf = queue.Queue(maxsize=12); lock = threading.Lock()

    def worker(seed):
        r = random.Random(seed); e = Encoder(AutoProcessor.from_pretrained(a.model))   # own tokenizer per thread
        while True:
            with lock: (sp, clip), qs = data.draw()
            qs = [shuffle_choices(q, r) for q in qs]
            try:
                qbuf.put(e(clip_images(sp, clip), qs, [q["answer"] for q in qs]))
            except Exception as e:                      # a bad example must not kill training
                print(f"[rank {rank}] skipped {clip}: {e}", flush=True)
    for i in range(int(os.environ.get("PREFETCH_THREADS", 3))):
        threading.Thread(target=worker, args=(rng.randrange(1 << 30),), daemon=True).start()
    model.train(); t0 = time.time(); run_loss = run_acc = run_n = 0.0
    while step < a.steps:
        for g in opt.param_groups: g["lr"] = a.lr * sched(step)
        for micro in range(a.accum):
            batch, pos = qbuf.get()
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            (out.loss / a.accum).backward()            # gradients averaged across ranks below (no DDP wrapper)
            with torch.no_grad():
                lab = batch["labels"][0]; lg = out.logits[0]
                idx = (lab != -100).nonzero().flatten()
                run_acc += float((lg[idx - 1].argmax(-1) == lab[idx]).float().mean()); run_loss += float(out.loss); run_n += 1
        if ddp:
            for p in params:
                if p.grad is None: continue
                if p.grad.device == dev: dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                else: g = p.grad.to(dev); dist.all_reduce(g, op=dist.ReduceOp.AVG); p.grad.copy_(g)
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True); step += 1
        if rank == 0 and step % 10 == 0:
            el = time.time() - t0
            print(f"step {step}/{a.steps} loss {run_loss / run_n:.4f} letter-acc {run_acc / run_n:.3f} "
                  f"lr {a.lr * sched(step):.2e} {el / 10:.1f}s/step", flush=True)
            t0 = time.time(); run_loss = run_acc = run_n = 0.0
        if rank == 0 and a.state_every and step % a.state_every == 0 and not (step % a.save_every == 0 or step == a.steps):
            os.makedirs(last, exist_ok=True)
            torch.save({"adapter": get_peft_model_state_dict(model), "opt": opt.state_dict(), "step": step},
                       os.path.join(last, "state.pt") + ".tmp")
            os.replace(os.path.join(last, "state.pt") + ".tmp", os.path.join(last, "state.pt"))
        if rank == 0 and (step % a.save_every == 0 or step == a.steps):
            os.makedirs(last, exist_ok=True)
            torch.save({"adapter": get_peft_model_state_dict(model), "opt": opt.state_dict(), "step": step},
                       os.path.join(last, "state.pt") + ".tmp")
            os.replace(os.path.join(last, "state.pt") + ".tmp", os.path.join(last, "state.pt"))
            model.save_pretrained(os.path.join(a.out, f"step_{step}"))
            json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"))
            print(f"saved step {step}", flush=True)
    if ddp: dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
