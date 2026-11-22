"""End-to-end distance regressor on FULL-RESOLUTION box crops (img_crops/<split>/<clip>/crops/<id>.jpg,
box enlarged 1.5x, 256 px) + calibration geometry (artifacts/<stem>_geo.npy, src/box3d_fit.geo).
DINOv3 ViT-B/16 fine-tuned (low lr) + geometry MLP -> log distance; L1 loss.
Train on train annotation samples (optionally + validation annotations); evaluate on validation distance
QUESTIONS (median relative error by range, accuracy with the real options and with fallback-style options).
-> <out>/pred_{validation,test}.json {question_id: dist}
usage: python src/dist_net.py <out dir> [--epochs 8] [--with-val-ann] [--cars-weight 2]
"""
import argparse, json, math, os, random, sys
import numpy as np
import torch, torch.nn as nn
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load(stem):
    S = [json.loads(l) for l in open(f"artifacts/{stem}.jsonl")]; G = np.load(f"artifacts/{stem}_geo.npy")
    items = []
    for i, s in enumerate(S):
        p = f"img_crops/{s['split']}/{s['clip']}/crops/{stem}_{i}.jpg"
        if os.path.isfile(p): items.append((p, G[i], s))
    return items


class DS(torch.utils.data.Dataset):
    def __init__(self, items, mu, sd, train):
        self.items, self.mu, self.sd, self.train = items, mu, sd, train

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        p, g, s = self.items[i]
        im = Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)
        if self.train:
            x = x * (0.8 + 0.4 * random.random()) + 0.1 * (random.random() - 0.5)       # brightness / offset jitter
            if random.random() < 0.5: x = x.flip(-1)
        x = (x.clamp(0, 1) - MEAN) / STD
        g = np.nan_to_num((g - self.mu) / self.sd); y = math.log(max(s.get("dist") or 1.0, 0.5))
        w = 1.0
        return x, torch.tensor(g, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(w)


class Net(nn.Module):
    def __init__(self, ngeo, model):
        super().__init__()
        from transformers import AutoModel
        self.bb = AutoModel.from_pretrained(model); d = self.bb.config.hidden_size
        self.nreg = int(getattr(self.bb.config, "num_register_tokens", 0))
        self.g = nn.Sequential(nn.Linear(ngeo, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * d + 128, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, x, g):
        h = self.bb(pixel_values=x).last_hidden_state
        f = torch.cat([h[:, 0], h[:, 1 + self.nreg:].mean(1), self.g(g)], 1)
        return self.head(f).squeeze(1)


def predict(net, items, mu, sd, bs=128):
    dl = torch.utils.data.DataLoader(DS(items, mu, sd, False), batch_size=bs, num_workers=8)
    out = []
    net.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, g, _, _ in dl: out.append(net(x.cuda(), g.cuda()).float().cpu())
    return np.exp(torch.cat(out).numpy())


def evaluate(items, pred):
    from dist_mono import pick
    t = np.array([float(s["choices"][s["answer"]].split()[0]) for _, _, s in items])
    cars = np.array([s["cat"] == "car" for _, _, s in items]); rel = np.abs(np.log(pred / t))
    acc = np.array([pick(p, s["choices"]) == s["answer"] for p, (_, _, s) in zip(pred, items)])
    rng = random.Random(0); fb = []
    for p, tt in zip(pred, t):                          # fallback-style options around the truth
        for _ in range(5):
            d = [tt] + [tt + x for x in rng.sample([8, 12, -6, -10], 3)]
            fb.append(min(range(4), key=lambda k: abs(math.log(max(d[k], .1) / p))) == 0)
    fb = np.array(fb).reshape(-1, 5).mean(1)
    for name, m in (("all", np.ones_like(cars)), ("cars", cars), ("cars >= 20 m", cars & (t >= 20)), ("cars 30-60 m", cars & (t >= 30) & (t < 60))):
        m = m.astype(bool)
        print(f"  {name:14s} n={m.sum():4d} median rel err {np.median(rel[m]):.3f} | acc real options {acc[m].mean():.3f} | acc fallback-style {fb[m].mean():.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--with-val-ann", action="store_true"); ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m"); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=-1, help=">= 0: seed python/numpy/torch (ensemble members)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.seed >= 0:
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    tr = [it for it in load("box3d_ann_train") if (it[2].get("dist") or 0) > 0.5]
    if a.with_val_ann: tr += [it for it in load("box3d_ann_validation") if (it[2].get("dist") or 0) > 0.5]
    G = np.stack([g for _, g, _ in tr]); mu, sd = np.nanmean(G, 0), np.nanstd(G, 0) + 1e-6
    vq = [it for it in load("box3d_q_validation") if it[2]["sub"] == "distance_from_2d_bbox"]
    tq = [it for it in load("box3d_q_test") if it[2]["sub"] == "distance_from_2d_bbox"]
    print(f"train {len(tr)} crops, validation questions {len(vq)}, test questions {len(tq)}", flush=True)
    net = Net(G.shape[1], a.model).cuda()
    opt = torch.optim.AdamW([{"params": net.bb.parameters(), "lr": a.lr},
                             {"params": list(net.g.parameters()) + list(net.head.parameters()), "lr": a.lr * 30}], weight_decay=0.01)
    dl = torch.utils.data.DataLoader(DS(tr, mu, sd, True), batch_size=a.bs, shuffle=True, num_workers=12, drop_last=True, persistent_workers=True)
    total = a.epochs * len(dl); step = 0; ep0 = 0
    ck = os.path.join(a.out, "last.pt")                # per-epoch resume (preemptible partitions)
    if os.path.isfile(ck):
        st = torch.load(ck, map_location="cpu", weights_only=False)
        net.load_state_dict(st["state"]); opt.load_state_dict(st["opt"]); ep0 = st["ep"] + 1; step = ep0 * len(dl)
        print(f"resumed after epoch {st['ep']}", flush=True)
    for ep in range(ep0, a.epochs):
        net.train(); run = 0.0
        for x, g, y, w in dl:
            f = min(1.0, (step + 1) / 300) * 0.5 * (1 + math.cos(math.pi * step / total))
            for i, grp in enumerate(opt.param_groups): grp["lr"] = (a.lr if i == 0 else a.lr * 30) * f
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = (net(x.cuda(), g.cuda()).float() - y.cuda()).abs().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            run += float(loss); step += 1
            if step % 200 == 0: print(f"ep {ep} step {step}/{total} L1(log) {run / 200:.4f}", flush=True); run = 0.0
        torch.save({"state": net.state_dict(), "opt": opt.state_dict(), "ep": ep}, ck + ".tmp"); os.replace(ck + ".tmp", ck)
        print(f"epoch {ep} done; validation questions:", flush=True)
        if not a.with_val_ann and (ep == a.epochs - 1 or ep % 2 == 1): evaluate(vq, predict(net, vq, mu, sd))
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "model": a.model}, os.path.join(a.out, "model.pt"))
    if os.path.isfile(ck): os.remove(ck)
    for name, its in (("validation", vq), ("test", tq)):
        p = predict(net, its, mu, sd)
        json.dump({s["qid"]: float(d) for d, (_, _, s) in zip(p, its)}, open(os.path.join(a.out, f"pred_{name}.json"), "w"))
    print("saved", flush=True)


if __name__ == "__main__":
    main()
