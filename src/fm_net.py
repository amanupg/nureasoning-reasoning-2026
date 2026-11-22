"""Future motion from FULL-RESOLUTION crops at two times: closing speed of the object in a stated box.
Inputs: key-frame crop (img_crops, box 1.5x) and the same region 1 s earlier (img_crops90, centred on the
key-frame box, 2.0x) through one shared DINOv3 ViT-B/16 (fine-tuned), + geometry (artifacts/<stem>_geo2.npy:
box, camera, category, calibration distances, stated x/y and range, ego velocity / acceleration).
Target: annotated closing_speed_mps (the exact source of the label); Huber loss.
Evaluation on validation future_motion questions: rule (0.5 m/s) accuracy, vs GBDT held-out and the SAM
tracker on the same questions; + averaging with the tracker's estimate.
-> <out>/pred_{validation,test}.json {question_id: closing speed}
usage: python src/fm_net.py <out dir> [--epochs 8] [--with-val-ann]
"""
import argparse, glob, json, math, os, random, sys
import numpy as np
import torch, torch.nn as nn
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
LAB = ("approaching ego", "moving away from ego", "maintaining relative distance")


def rule(c, t=0.5):
    return LAB[0] if c > t else (LAB[1] if c < -t else LAB[2])


def load(stem, sub=None):
    S = [json.loads(l) for l in open(f"artifacts/{stem}.jsonl")]; G = np.load(f"artifacts/{stem}_geo2.npy")
    items = []
    for i, s in enumerate(S):
        if sub and s.get("sub") != sub: continue
        a = f"img_crops/{s['split']}/{s['clip']}/crops/{stem}_{i}.jpg"; b = f"img_crops90/{s['split']}/{s['clip']}/crops/{stem}_{i}.jpg"
        if os.path.isfile(a) and os.path.isfile(b): items.append((a, b, G[i], s))
    return items


def img(p, train, flip):
    x = torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR), np.float32) / 255).permute(2, 0, 1)
    if flip: x = x.flip(-1)
    return (x - MEAN) / STD


class DS(torch.utils.data.Dataset):
    def __init__(self, items, mu, sd, train):
        self.items, self.mu, self.sd, self.train = items, mu, sd, train

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        a, b, g, s = self.items[i]
        flip = self.train and random.random() < 0.5            # same flip for both frames (motion preserved)
        g = np.nan_to_num((g - self.mu) / self.sd); y = float(np.clip(s.get("closing") or 0.0, -20, 20))
        return img(a, self.train, flip), img(b, self.train, flip), torch.tensor(g, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class Net(nn.Module):
    def __init__(self, ngeo, model):
        super().__init__()
        from transformers import AutoModel
        self.bb = AutoModel.from_pretrained(model); d = self.bb.config.hidden_size
        self.nreg = int(getattr(self.bb.config, "num_register_tokens", 0))
        self.g = nn.Sequential(nn.Linear(ngeo, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.head = nn.Sequential(nn.Linear(6 * d + 128, 384), nn.GELU(), nn.Linear(384, 1))

    def emb(self, x):
        h = self.bb(pixel_values=x).last_hidden_state
        return torch.cat([h[:, 0], h[:, 1 + self.nreg:].mean(1)], 1)

    def forward(self, a, b, g):
        ea, eb = self.emb(a), self.emb(b)
        return self.head(torch.cat([ea, eb, ea - eb, self.g(g)], 1)).squeeze(1)


def predict(net, items, mu, sd, bs=96):
    dl = torch.utils.data.DataLoader(DS(items, mu, sd, False), batch_size=bs, num_workers=8)
    out = []; net.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for a, b, g, _ in dl: out.append(net(a.cuda(), b.cuda(), g.cuda()).float().cpu())
    return torch.cat(out).numpy()


def evaluate(items, pred):
    gp = "artifacts/gbdt_heldout_preds.jsonl"
    G = {json.loads(l)["question_id"]: json.loads(l) for l in open(gp)} if os.path.isfile(gp) else {}
    trk = {}
    for f in glob.glob("runs_fm/sam21_full_s*of4.json"): trk.update(json.load(open(f)))
    gold = [s["choices"][s["answer"]] for _, _, _, s in items]
    acc = np.array([rule(p) == y for p, y in zip(pred, gold)])
    print(f"  all fm questions n={len(items)} net rule accuracy {acc.mean():.3f}", flush=True)
    both = [i for i, (_, _, _, s) in enumerate(items) if s["qid"] in G]
    if both: print(f"  GBDT held-out subset n={len(both)}: net {acc[both].mean():.3f} GBDT {np.mean([G[items[i][3]['qid']]['model'] == G[items[i][3]['qid']]['gold'] for i in both]):.3f}", flush=True)
    try:
        import camgeo, fm_rule as F
        rig = {}; te = []
        for i, (_, _, _, s) in enumerate(items):
            r = trk.get(s["qid"])
            if not r or r.get("missing"): continue
            if s["clip"] not in rig: rig[s["clip"]] = camgeo.Rig(json.load(open(f"cache/validation/{s['clip']}/metadata.json")))
            e = F.closing(r, rig[s["clip"]], "h", edge=3, min_frames=2)[0]
            if np.isfinite(e): te.append((i, e))
        if te:
            ii = [i for i, _ in te]; e = np.array([x for _, x in te])
            tr_acc = np.mean([rule(x) == gold[i] for i, x in te])
            avg = np.mean([rule(0.5 * x + 0.5 * pred[i]) == gold[i] for i, x in te])
            print(f"  tracker-usable n={len(te)}: net {acc[ii].mean():.3f} tracker {tr_acc:.3f} average(net, tracker) {avg:.3f}", flush=True)
    except Exception as ex:
        print("  (tracker comparison skipped:", ex, ")", flush=True)


def holdout_report(items, pred, tag):
    """Annotated-closing-speed rule accuracy on held-out train annotations (test-like years)."""
    t = np.array([s["closing"] for _, _, _, s in items]); acc = np.mean([rule(p) == rule(x) for p, x in zip(pred, t)])
    print(f"  {tag} n={len(items)}: rule acc {acc:.3f}, MAE {np.mean(np.abs(pred - t)):.2f} m/s, "
          f"pred |c|<0.5 {np.mean(np.abs(pred) < .5):.2f} (truth {np.mean(np.abs(t) < .5):.2f})", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--with-val-ann", action="store_true"); ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m"); ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--seed", type=int, default=-1, help=">= 0: seed python/numpy/torch")
    ap.add_argument("--holdout-years", default="", help="e.g. 2024,2025: train clips from these log years are held out "
                    "(test is 93%% 2025 logs, validation 100%% 2026) and scored as a test-like check")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.seed >= 0:
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    tr = [it for it in load("box3d_ann_train") if it[3].get("closing") is not None]
    yrs = tuple(y for y in a.holdout_years.split(",") if y)
    ho = [it for it in tr if yrs and it[3]["clip"][:4] in yrs]; tr = [it for it in tr if not (yrs and it[3]["clip"][:4] in yrs)]
    if ho:   # one pair per object track is plenty for a check; cap for speed
        random.Random(0).shuffle(ho); ho = ho[:6000]
    if a.with_val_ann: tr += [it for it in load("box3d_ann_validation") if it[3].get("closing") is not None]
    G = np.stack([g for _, _, g, _ in tr]); mu, sd = np.nanmean(G, 0), np.nanstd(G, 0) + 1e-6
    vq = load("box3d_q_validation", "future_motion_label"); tq = load("box3d_q_test", "future_motion_label")
    print(f"train {len(tr)} pairs, held-out ({a.holdout_years or '-'}) {len(ho)}, validation fm questions {len(vq)}, test fm questions {len(tq)}", flush=True)
    net = Net(G.shape[1], a.model).cuda()
    opt = torch.optim.AdamW([{"params": net.bb.parameters(), "lr": a.lr},
                             {"params": list(net.g.parameters()) + list(net.head.parameters()), "lr": a.lr * 30}], weight_decay=0.01)
    dl = torch.utils.data.DataLoader(DS(tr, mu, sd, True), batch_size=a.bs, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True)
    total = a.epochs * len(dl); step = 0; ep0 = 0; huber = nn.HuberLoss(delta=1.0)
    ck = os.path.join(a.out, "last.pt")                # per-epoch resume (preemptible partitions)
    if os.path.isfile(ck):
        st = torch.load(ck, map_location="cpu", weights_only=False)
        net.load_state_dict(st["state"]); opt.load_state_dict(st["opt"]); ep0 = st["ep"] + 1; step = ep0 * len(dl)
        print(f"resumed after epoch {st['ep']}", flush=True)
    for ep in range(ep0, a.epochs):
        net.train(); run = 0.0
        for x1, x2, g, y in dl:
            f = min(1.0, (step + 1) / 300) * 0.5 * (1 + math.cos(math.pi * step / total))
            for i, grp in enumerate(opt.param_groups): grp["lr"] = (a.lr if i == 0 else a.lr * 30) * f
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = huber(net(x1.cuda(), x2.cuda(), g.cuda()).float(), y.cuda())
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            run += float(loss); step += 1
            if step % 200 == 0: print(f"ep {ep} step {step}/{total} huber {run / 200:.4f}", flush=True); run = 0.0
        torch.save({"state": net.state_dict(), "opt": opt.state_dict(), "ep": ep}, ck + ".tmp"); os.replace(ck + ".tmp", ck)
        print(f"epoch {ep} done", flush=True)
        if ep == a.epochs - 1 or ep % 2 == 1:
            if not a.with_val_ann: print(" validation fm questions:", flush=True); evaluate(vq, predict(net, vq, mu, sd))
            if ho: holdout_report(ho, predict(net, ho, mu, sd), f"held-out train {a.holdout_years}")
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "model": a.model}, os.path.join(a.out, "model.pt"))
    if os.path.isfile(ck): os.remove(ck)
    for name, its in (("validation", vq), ("test", tq)):
        p = predict(net, its, mu, sd)
        json.dump({s["qid"]: float(c) for c, (_, _, _, s) in zip(p, its)}, open(os.path.join(a.out, f"pred_{name}.json"), "w"))
    if ho:
        p = predict(net, ho, mu, sd)
        json.dump([{"clip": s["clip"], "closing": s["closing"], "pred": float(c)} for c, (_, _, _, s) in zip(p, ho)], open(os.path.join(a.out, "pred_holdout.json"), "w"))
    print("saved", flush=True)


if __name__ == "__main__":
    main()
