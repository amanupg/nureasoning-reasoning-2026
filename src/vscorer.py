"""Vision trajectory scorer (Hydra-MDP-style) for nuReasoning NPS.

Each of the 63 vocabulary candidates is a query (its ego-frame trajectory +
vocab params) that cross-attends to frozen DINOv3 scene tokens (2 frames x 8
cams x [CLS + 6x8 grid]) plus an ego/route token. Separate heads predict each
sub-metric (never the aggregate: distilling the aggregate measured worse,
Hydra-MDP 80.2 vs 83.0). Selection = log-linear combination with weights tuned
on held-out TRAIN clips, optionally inside a trust region.

--no-vision zeroes the scene tokens: the same model with the cameras removed.
That ablation is what shows whether vision is doing the work.
"""
import argparse, json, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

SUBS = ["col", "da", "prog", "comf", "hum"]


def _load_cands(split, cands):
    """legacy: artifacts/cands_{split}.npz (63 candidates, vocab [scale, lat]).
    v148: merge artifacts/labels148/{split}_*.npz (vocab [is_abs, value, lat])."""
    if cands == "legacy":
        z = np.load(f"artifacts/cands_{split}.npz", allow_pickle=False)
        return {k: z[k] for k in z.files}
    import glob
    parts = [np.load(f) for f in sorted(glob.glob(f"artifacts/{cands}/{split}_[0-9]*.npz"))]  # key-frame shards only (not _kf*)
    assert parts, f"no shards for {split} in artifacts/{cands}"
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files if k != "vocab"}
    out["vocab"] = parts[0]["vocab"]
    for p in parts: assert np.array_equal(p["vocab"], out["vocab"]), "shards disagree on vocabulary"
    return out


def default_index(vocab):
    if vocab.shape[1] == 2:
        return int(np.where((vocab[:, 0] == 1.0) & (vocab[:, 1] == 0.0))[0][0])
    return int(np.where((vocab[:, 0] == 0) & (vocab[:, 1] == 1.0) & (vocab[:, 2] == 0.0))[0][0])


def load_split(split, feat_root, cands="legacy"):
    z = _load_cands(split, cands)
    names = [str(n) for n in z["names"]]
    keep = [i for i, n in enumerate(names) if os.path.isfile(os.path.join(feat_root, split, n + ".npy"))]
    d = {"names": [names[i] for i in keep], "traj": z["traj"][keep].astype(np.float32),
         "feat": z["feat"][keep], "vocab": z["vocab"].astype(np.float32)}
    if "labels" in z: d["labels"] = z["labels"][keep]
    if d["vocab"].shape[1] == 3:                   # absolute arcs are metres (0..77): bring to ~[0, 1.5]
        d["vocab"][:, 1] = np.where(d["vocab"][:, 0] == 1, d["vocab"][:, 1] / 50.0, d["vocab"][:, 1])
    d["scene"] = np.stack([np.load(os.path.join(feat_root, split, n + ".npy")) for n in d["names"]])
    return d


class Scorer(nn.Module):
    def __init__(self, dfeat=768, nf=27, d=256, layers=3, heads=8, dv=2):
        super().__init__()
        self.scene_in = nn.Linear(dfeat, d)
        self.e_frame = nn.Parameter(torch.zeros(2, 1, 1, d)); self.e_cam = nn.Parameter(torch.zeros(1, 8, 1, d))
        self.e_tok = nn.Parameter(torch.zeros(1, 1, 49, d))
        for p in (self.e_frame, self.e_cam, self.e_tok): nn.init.normal_(p, std=0.02)
        self.ego = nn.Sequential(nn.Linear(2 * nf, d), nn.GELU(), nn.Linear(d, d))
        self.q = nn.Sequential(nn.Linear(11 * 4 + dv, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerDecoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, layers)
        self.h_col = nn.Linear(d, 3); self.h_da = nn.Linear(d, 1); self.h_comf = nn.Linear(d, 1)
        self.h_prog = nn.Linear(d, 1); self.h_hum = nn.Linear(d, 1)

    def forward(self, scene, feat, traj, vocab):
        B = scene.shape[0]
        s = self.scene_in(scene)                                  # [B, 2, 8, 49, d]
        s = s + self.e_frame.view(1, 2, 1, 1, -1) + self.e_cam.view(1, 1, 8, 1, -1) + self.e_tok.view(1, 1, 1, 49, -1)
        s = s.reshape(B, 2 * 8 * 49, -1)
        e = self.ego(feat).unsqueeze(1)
        mem = torch.cat([e, s], 1)
        x, y, yaw = traj[..., 0] / 30.0, traj[..., 1] / 5.0, traj[..., 2]
        tq = torch.stack([x, y, torch.sin(yaw), torch.cos(yaw)], -1).flatten(2)
        q = self.q(torch.cat([tq, vocab.unsqueeze(0).expand(B, -1, -1)], -1)) + e
        h = self.dec(q, mem)
        return {"col": self.h_col(h), "da": self.h_da(h).squeeze(-1), "comf": self.h_comf(h).squeeze(-1),
                "prog": self.h_prog(h).squeeze(-1), "hum": self.h_hum(h).squeeze(-1)}


def probs(o):
    pc = F.softmax(o["col"], -1)
    return {"col": (pc * torch.tensor([0., .5, 1.], device=pc.device)).sum(-1), "da": torch.sigmoid(o["da"]),
            "comf": torch.sigmoid(o["comf"]), "prog": torch.sigmoid(o["prog"]), "hum": torch.sigmoid(o["hum"])}


def loss_fn(o, y):
    col_t = torch.round(y[..., 0] * 2).long()
    return (F.cross_entropy(o["col"].flatten(0, 1), col_t.flatten())
            + F.binary_cross_entropy_with_logits(o["da"], (y[..., 1] > .5).float())
            + 0.2 * F.binary_cross_entropy_with_logits(o["comf"], (y[..., 3] > .5).float())
            + F.binary_cross_entropy_with_logits(o["prog"], y[..., 2])
            + F.binary_cross_entropy_with_logits(o["hum"], y[..., 4]))


def nps(y):
    return y[..., 0] * y[..., 1] * (0.3 * y[..., 2] + 0.2 * y[..., 3] + 0.5 * y[..., 4])


def select_score(p, w):
    e = 1e-6
    return w[0] * np.log(p["col"] + e) + w[1] * np.log(p["da"] + e) + \
        np.log(0.3 * p["prog"] + 0.2 * p["comf"] + 0.5 * p["hum"] + e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train"); ap.add_argument("--eval", default="validation")
    ap.add_argument("--feat-root", default="feat/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--train-frac", type=float, default=1.0, help="use only this fraction of eval split for training (debug)")
    ap.add_argument("--out", default="runs_v"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cands", default="legacy", help="legacy | labels148")
    ap.add_argument("--predict", default="", help="also write per-clip selections for this split (e.g. test)")
    ap.add_argument("--dump-preds", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"
    if a.train == a.eval:                                   # debug mode: split one set by clip
        D = load_split(a.eval, a.feat_root, a.cands); n = len(D["names"]); perm = np.random.permutation(n)
        cut = int(n * a.train_frac)
        tr = {k: (v[perm[:cut]] if isinstance(v, np.ndarray) and k != "vocab" else v) for k, v in D.items() if k != "names"}
        ev = {k: (v[perm[cut:]] if isinstance(v, np.ndarray) and k != "vocab" else v) for k, v in D.items() if k != "names"}
    else:
        tr, ev = load_split(a.train, a.feat_root, a.cands), load_split(a.eval, a.feat_root, a.cands)
    print(f"train clips {len(tr['traj'])}  eval clips {len(ev['traj'])}  no_vision={a.no_vision}", flush=True)

    fm = np.nanmean(tr["feat"], 0); fs = np.nanstd(tr["feat"], 0) + 1e-6
    def fe(f):
        z = (f - fm) / fs; m = np.isnan(z); z[m] = 0
        return np.concatenate([z, m.astype(np.float32)], -1).astype(np.float32)
    vocab = torch.tensor(ev["vocab"], device=dev)

    # hold out 15% of TRAIN clips to tune selection weights (never the eval split)
    ntr = len(tr["traj"]); pm = np.random.permutation(ntr); cal = pm[:int(.15 * ntr)]; fit = pm[int(.15 * ntr):]

    def to_t(d, idx):
        sc = torch.tensor(d["scene"][idx], dtype=torch.float16)
        if a.no_vision: sc = torch.zeros_like(sc)
        return (sc, torch.tensor(fe(d["feat"][idx])), torch.tensor(d["traj"][idx]),
                torch.tensor(d["labels"][idx]) if "labels" in d else None)

    NV, DV = ev["vocab"].shape
    model = Scorer(dv=DV).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * max(1, len(fit) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)

    def predict(d, idx):
        model.eval(); out = {k: [] for k in SUBS}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(idx), 64):
                sc, f, t, _ = to_t(d, idx[i:i + 64])
                p = probs(model(sc.to(dev).float(), f.to(dev), t.to(dev), vocab))
                for k in SUBS: out[k].append(p[k].float().cpu().numpy())
        model.train(); return {k: np.concatenate(v) for k, v in out.items()}

    t0 = time.time()
    for ep in range(a.epochs):
        np.random.shuffle(fit); tot = 0
        for i in range(0, len(fit) - a.bs + 1, a.bs):
            sc, f, t, y = to_t(tr, fit[i:i + a.bs])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l = loss_fn(model(sc.to(dev).float(), f.to(dev), t.to(dev), vocab), y.to(dev))
            opt.zero_grad(set_to_none=True); l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tot += l.item()
        if ep % 5 == 4 or ep == a.epochs - 1:
            print(f"  epoch {ep+1:3d}  loss {tot/max(1,len(fit)//a.bs):.4f}  {time.time()-t0:.0f}s", flush=True)

    vs = ev["vocab"]; cur = default_index(vs)
    if DV == 2:
        sc_, la_ = vs[:, 0], vs[:, 1]; rel = np.ones(NV, bool)
    else:
        rel = vs[:, 0] == 0; sc_ = np.where(rel, vs[:, 1], 99.0); la_ = vs[:, 2]
    regions = {"all": np.ones(NV, bool), "tr_.3_1": rel & (abs(sc_ - 1) <= .3) & (abs(la_) <= 1),
               "tr_.45_1": rel & (abs(sc_ - 1) <= .45) & (abs(la_) <= 1), "tr_.6_2": rel & (abs(sc_ - 1) <= .6) & (abs(la_) <= 2)}
    if DV == 3:
        regions["abs_only"] = ~rel
    pc = predict(tr, cal); Nc = nps(tr["labels"][cal])
    best = None
    for rn, ok in regions.items():
        for wn in (0, .25, .5, 1, 2, 4, 8):
            for wd in (0, .25, .5, 1, 2, 4, 8):
                s = np.where(ok[None], select_score(pc, (wn, wd)), -1e9)
                v = Nc[np.arange(len(Nc)), s.argmax(1)].mean()
                if best is None or v > best[0]: best = (v, rn, (wn, wd))
    idx = np.arange(len(ev["traj"])); pe = predict(ev, idx); Ne = nps(ev["labels"])
    if a.dump_preds:
        os.makedirs(a.out, exist_ok=True)
        np.savez_compressed(f"{a.out}/preds_{'novision' if a.no_vision else 'vision'}.npz",
                            calib_names=np.array([tr["names"][i] for i in cal]) if "names" in tr else np.array([]),
                            eval_names=np.array(ev["names"]) if "names" in ev else np.array([]),
                            **{f"calib_{k}": v for k, v in pc.items()}, **{f"eval_{k}": v for k, v in pe.items()})
    sel = np.where(regions[best[1]][None], select_score(pe, best[2]), -1e9).argmax(1)
    from scipy.stats import spearmanr
    rho = np.nanmean([spearmanr(select_score(pe, best[2])[c], Ne[c]).statistic
                      for c in range(len(Ne)) if np.unique(Ne[c]).size > 1])
    res = {"no_vision": a.no_vision, "train_clips": int(ntr), "eval_clips": int(len(Ne)),
           "default": float(Ne[:, cur].mean()), "selected": float(Ne[idx, sel].mean()),
           "oracle": float(Ne.max(1).mean()), "spearman": float(rho), "region": best[1], "w": best[2]}
    res["headroom_captured_pct"] = 100 * (res["selected"] - res["default"]) / (res["oracle"] - res["default"])
    print(json.dumps(res, indent=1), flush=True)
    os.makedirs(a.out, exist_ok=True)
    tag = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + ("_novision" if a.no_vision else "_vision")
    if a.predict:
        P = load_split(a.predict, a.feat_root, a.cands)
        assert np.array_equal(P["vocab"], ev["vocab"]), "prediction vocabulary differs from training"
        pp = predict(P, np.arange(len(P["traj"])))
        if a.dump_preds:
            np.savez_compressed(f"{a.out}/preds_{'novision' if a.no_vision else 'vision'}_{a.predict}.npz",
                                names=np.array(P["names"]), **pp)
        pick = np.where(regions[best[1]][None], select_score(pp, best[2]), -1e9).argmax(1)
        sel = {n: int(k) for n, k in zip(P["names"], pick)}
        json.dump({"split": a.predict, "cands": a.cands, "region": best[1], "w": best[2],
                   "default_index": int(cur), "selections": sel,
                   "changed_from_default": int(sum(k != cur for k in pick))},
                  open(f"{a.out}/selections_{a.predict}_{tag}.json", "w"))
        print(f"wrote {len(sel)} {a.predict} selections ({sum(k != cur for k in pick)} differ from default)", flush=True)
    torch.save({"model": model.state_dict(), "fm": fm, "fs": fs, "res": res, "args": vars(a)},
               f"{a.out}/vscorer_{tag}.pt")
    json.dump(res, open(f"{a.out}/vscorer_{tag}.json", "w"))


if __name__ == "__main__":
    main()
