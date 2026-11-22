"""FULL fine-tune (all weights, language model + image encoder) of a Qwen3.5-family VLM on generated
multiple-choice questions, sharded with torch FSDP2 (fully_shard; fp32 master weights, bf16 compute, fp32 reduce).
torchrun --nproc_per_node N src/vlm_sft_full.py --model <dir> --out <dir> [--init_adapter <LoRA dir>] [--steps --accum --lr]
- 2026-11-15 (limitations audit): the LoRA runs (src/vlm_sft.py) never trained all weights; this is the full-FT path.
- warm start: base + --init_adapter LoRA merged (run 1 = runs_vlm/qd4b_r64/step_3000), then every weight trainable
- same data path as vlm_sft.py (DENSE=1 -> vlm_data_dense (clip, frame) examples), same prompts, options re-lettered
  per draw, loss on the answer letters only (logits computed only at the letter positions: no full-vocab logits)
- resumable: per-rank local shards of model + optimizer + step in <out>/last/ (atomic swap; same world size); HF bf16 snapshots <out>/step_<n>/
  loadable by src/vlm_infer.py with no adapter (processor/tokenizer files copied from the base dir)
"""
import argparse, json, math, os, random, shutil, time
import torch, torch.distributed as dist
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, HERE)
from vlm_data import Encoder, TrainSet, clip_images
if os.environ.get("DENSE") == "1":
    from vlm_data_dense import TrainSet, clip_images
from vlm_sft import shuffle_choices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--accum", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--wd", type=float, default=0.1, help="AdamW weight decay (leader full-FT recipe: 0.1)")
    ap.add_argument("--vision_lr_mult", type=float, default=float(os.environ.get("VISION_LR_MULT", "1.0")),
                    help="lr multiplier for the image encoder + merger (Qwen-VL practice: 0.1-0.2)")
    ap.add_argument("--save_every", type=int, default=500); ap.add_argument("--state_every", type=int, default=25)
    ap.add_argument("--split", default="train"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init_adapter", default="", help="LoRA adapter merged into the base before training (ignored on resume)")
    ap.add_argument("--freeze_patch_embed", type=int, default=0)
    ap.add_argument("--reduce_fp32", type=int, default=0, help="1: fp32 gradient reduce + accumulation (default bf16: half the traffic)")
    a = ap.parse_args()
    dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size()
    lr_ = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(lr_); dev = torch.device("cuda", lr_)
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    proc = AutoProcessor.from_pretrained(a.model); enc = Encoder(proc)
    import struct
    with open(os.path.join(a.model, "model.safetensors"), "rb") as fh:       # base key names + dtypes (header only)
        hdr = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
    base_dt = {k: (torch.float32 if v["dtype"] == "F32" else torch.bfloat16) for k, v in hdr.items() if k != "__metadata__"}
    last = os.path.join(a.out, "last"); resuming = os.path.isfile(os.path.join(last, "step.json"))
    t_load = time.time()
    # straight onto the GPU (no per-rank CPU copy: 6 ranks x 9 GB of host RAM pushed 13915037 to its 96 GB cgroup limit)
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": dev})
    if a.init_adapter and not resuming:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.init_adapter).merge_and_unload()
        if rank == 0: print(f"warm start: merged {a.init_adapter}", flush=True)
    model.config.use_cache = False
    model.to(torch.float32)                                   # fp32 master weights (sharded below)
    for p in model.parameters(): p.requires_grad_(True)
    if a.freeze_patch_embed:
        for p in model.model.visual.patch_embed.parameters(): p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # 2026-11-15: A6000 nodes are PCIe (NVLink only within GPU pairs): the default (re-gather every layer in forward AND
    # backward of EVERY micro-step) left the GPUs waiting on NCCL (13915037: 55 s/step at accum 2, ~120 W). Keep the bf16
    # weights gathered from the first forward of a step until the last backward: one all-gather + one reduce-scatter per step.
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32 if a.reduce_fp32 else torch.bfloat16)
    for blk in model.model.visual.blocks: fully_shard(blk, mp_policy=mp, reshard_after_forward=False)
    for lyr in model.model.language_model.layers: fully_shard(lyr, mp_policy=mp, reshard_after_forward=False)
    fully_shard(model, mp_policy=mp, reshard_after_forward=False)                          # root: embeddings + tied lm_head, merger, patch embed, norms
    torch.cuda.empty_cache(); import gc, ctypes; gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass
    n_par = sum(p.numel() for p in model.parameters())         # DTensor numel = global size
    if rank == 0: print(f"FSDP2 full FT: {n_par / 1e9:.2f}B params over {world} ranks, load+shard {time.time() - t_load:.0f}s, "
                        f"mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    # 2026-11-15: image encoder + merger at lr * vision_lr_mult (default 1.0 = one rate for everything, as before)
    vis = [p for n, p in model.named_parameters() if p.requires_grad and ".visual." in f".{n}."]
    vid = {id(p) for p in vis}; lang = [p for p in params if id(p) not in vid]
    # 2026-11-15: no weight decay on 1-D params (norm gains, biases), as in HF Trainer / ms-swift (the leader's stack)
    groups = []
    for ps, mult in ((lang, 1.0), (vis, a.vision_lr_mult)):
        for decay in (True, False):
            g = [p for p in ps if (p.dim() >= 2) == decay]
            if g: groups.append({"params": g, "mult": mult, "weight_decay": a.wd if decay else 0.0})
    if rank == 0:
        for g in groups: print(f"param group: x{g['mult']} wd {g['weight_decay']} {sum(p.numel() for p in g['params']) / 1e9:.4f}B "
                               f"({len(g['params'])} tensors)", flush=True)
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.99))
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))
    step = 0
    from torch.distributed.tensor import DTensor
    named = list(model.named_parameters())                    # sharded params (DTensor), tied lm_head listed once
    def loc(t): return t.to_local() if isinstance(t, DTensor) else t      # under no_grad: the local shard itself

    def checksum():
        with torch.no_grad(): v = torch.stack([loc(p).float().sum() for _, p in named[:50]]).sum()
        dist.all_reduce(v); return float(v)

    if resuming:
        # per-rank local shards (see save_state); needs the same world size
        st = torch.load(os.path.join(last, f"rank{rank}.pt"), map_location=dev, weights_only=False)
        assert st["world"] == world, f"checkpoint written by {st['world']} ranks, now {world}"
        with torch.no_grad():
            for n, p_ in named: loc(p_).copy_(st["model"][n])
        for i, p_ in enumerate(params):
            if i not in st["optim"]: continue
            opt.state[p_] = {k: (v.cpu() if v.dim() == 0 else
                                 DTensor.from_local(v, p_.device_mesh, p_.placements, shape=p_.shape, stride=p_.stride())
                                 if isinstance(p_, DTensor) else v) if torch.is_tensor(v) else v
                             for k, v in st["optim"][i].items()}
        step = st["step"]; del st; torch.cuda.empty_cache()
        cs = checksum()
        if rank == 0: print(f"resumed at step {step} (checksum {cs:.6e})", flush=True)

    def save_state():
        # 2026-11-15: one torch.save per rank of its LOCAL shards (host copies one storage at a time). DCP's save
        # staged ~50 GB of host RAM across 6 ranks and OOM-killed smoke 13915037 at its first state save.
        tmp = last + ".tmp"
        if rank == 0:
            if os.path.isdir(tmp): shutil.rmtree(tmp)
            os.makedirs(tmp)
        dist.barrier()
        with torch.no_grad():
            st = {"world": world, "step": step, "model": {n: loc(p_) for n, p_ in named},
                  "optim": {i: {k: (loc(v) if torch.is_tensor(v) else v) for k, v in opt.state[p_].items()}
                            for i, p_ in enumerate(params) if p_ in opt.state}}
            torch.save(st, os.path.join(tmp, f"rank{rank}.pt")); del st
        cs = checksum(); dist.barrier()
        if rank == 0:
            json.dump({"step": step, "world": world, "checksum": cs}, open(os.path.join(tmp, "step.json"), "w"))
            old = last + ".old"
            if os.path.isdir(last): os.replace(last, old)
            os.replace(tmp, last)
            if os.path.isdir(old): shutil.rmtree(old)
            print(f"state saved at step {step} (checksum {cs:.6e})", flush=True)
        dist.barrier()

    def save_hf(path):
        # gathered one tensor at a time (collective), kept on rank 0 in the base checkpoint's dtype: ~9 GB host RAM
        sd = {}
        with torch.no_grad():
            for n, p_ in named:
                full = p_.full_tensor() if isinstance(p_, DTensor) else p_
                if rank == 0 and n in base_dt: sd[n] = full.to(base_dt[n]).cpu()
                del full
            if rank == 0:
                for n, b in model.named_buffers():
                    if n in base_dt and n not in sd: sd[n] = b.to(base_dt[n]).cpu()
        if rank == 0:
            from safetensors.torch import save_file
            extra = sorted({n for n, _ in named} - set(base_dt) - {"lm_head.weight"})
            if extra: print(f"[save] {len(extra)} params not in the base (not saved), e.g. {extra[:3]}", flush=True)
            miss = set(base_dt) - set(sd)
            assert not miss, f"snapshot lacks {len(miss)} base keys, e.g. {sorted(miss)[:3]}"
            sd = {k: v.contiguous() for k, v in sd.items()}
            tmp = path + ".tmp"; os.makedirs(tmp, exist_ok=True)
            save_file(sd, os.path.join(tmp, "model.safetensors"), metadata={"format": "pt"})
            for f in os.listdir(a.model):                       # config, processor, tokenizer, chat template files
                src = os.path.join(a.model, f)
                if os.path.isfile(src) and not f.endswith(".safetensors") and not f.endswith(".index.json"):
                    shutil.copy(src, tmp)
            if os.path.isdir(path): shutil.rmtree(path)
            os.replace(tmp, path)
            json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"))
            print(f"saved step {step} -> {path}", flush=True)
        del sd
        dist.barrier()

    data = TrainSet(a.split, seed=a.seed * 1000 + rank + 7919 * step); rng = random.Random(a.seed * 31 + rank + step)
    import queue, threading
    qbuf = queue.Queue(maxsize=12); lock = threading.Lock()

    def worker(seed):
        r = random.Random(seed); e = Encoder(AutoProcessor.from_pretrained(a.model))
        while True:
            with lock: (sp, clip), qs = data.draw()
            qs = [shuffle_choices(q, r) for q in qs]
            try:
                qbuf.put(e(clip_images(sp, clip), qs, [q["answer"] for q in qs]))
            except Exception as ex:
                print(f"[rank {rank}] skipped {clip}: {ex}", flush=True)
    for i in range(int(os.environ.get("PREFETCH_THREADS", 3))):
        threading.Thread(target=worker, args=(rng.randrange(1 << 30),), daemon=True).start()
    model.train(); t0 = time.time(); run_loss = run_acc = run_n = 0.0; s0 = step
    s_print = step; peak_k = {}                                              # peak GPU memory (GiB) by questions per example k
    while step < a.steps:
        for g in opt.param_groups: g["lr"] = a.lr * g.get("mult", 1.0) * sched(step)
        for micro in range(a.accum):
            fin = micro == a.accum - 1
            model.set_requires_gradient_sync(fin); model.set_reshard_after_backward(fin)
            batch, pos = qbuf.get(); torch.cuda.reset_peak_memory_stats()
            labels = batch.pop("labels")[0]
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            pos_t = torch.tensor(pos, device=dev)
            out = model(**batch, logits_to_keep=pos_t)
            lg = out.logits[0].float(); tgt = labels[torch.tensor(pos) + 1].to(dev)
            loss = F.cross_entropy(lg, tgt)
            (loss / a.accum).backward()
            with torch.no_grad():
                run_acc += float((lg.argmax(-1) == tgt).float().mean()); run_loss += float(loss); run_n += 1
            k = len(pos); peak_k[k] = max(peak_k.get(k, 0.0), torch.cuda.max_memory_allocated() / 2**30)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % 10 == 0 or step - s0 <= 3:
            gn = gn.full_tensor() if hasattr(gn, "full_tensor") else gn
            if rank == 0:
                el = time.time() - t0; ns = step - s_print       # steps since the last print (incl. any saves)
                print(f"step {step}/{a.steps} loss {run_loss / run_n:.4f} letter-acc {run_acc / run_n:.3f} "
                      f"lr {a.lr * sched(step):.2e} {el / max(1, ns):.1f}s/step gnorm {float(gn):.3f} "
                      f"peakmem {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB", flush=True)
            t0 = time.time(); run_loss = run_acc = run_n = 0.0; s_print = step
        if step % a.save_every == 0 or step == a.steps:
            save_state(); save_hf(os.path.join(a.out, f"step_{step}"))
        elif a.state_every and step % a.state_every == 0:
            save_state()
    KMAX = 16                                                  # per-rank peak by k (k <= KMAX), 0 = k not drawn
    pk = torch.tensor([peak_k.get(k, 0.0) for k in range(1, KMAX + 1)], device=dev)
    allp = [torch.zeros_like(pk) for _ in range(world)]; dist.all_gather(allp, pk)
    if rank == 0:
        for k in range(1, KMAX + 1):
            v = [round(float(x[k - 1]), 1) for x in allp]
            if any(v): print(f"peak GiB per rank at k={k} (micro-batch, fwd+bwd; 0 = not drawn on that rank): {v}", flush=True)
        print("peak GiB per rank (any k):", [round(float(x.max()), 1) for x in allp], flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
