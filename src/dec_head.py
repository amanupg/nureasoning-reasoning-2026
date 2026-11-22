"""Driving-decision head: DINOv3 scene tokens + ego kinematics -> (longitudinal,
lateral) decision, supervised by the train annotations' "Driving decision".

Oracle decisions add +0.035 to the chooser (validation). This measures how much
of that a camera model recovers. Train predictions are OUT-OF-FOLD (5 folds split
by drive log) so the chooser is trained on honest predictions; validation/test
use the average of the 5 fold models. Output uses the percfeat layout:
artifacts/decpred_<split>.npz (clip = [p_lon..., p_lat...]).
With --no-vision the same model sees kinematics only (the control).
"""
import argparse, collections, glob, json, os, random, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

LON_MAP = {"Gently accelerate (speed up)": "Gently accelerate", "Quickly accelerate (speed up)": "Quickly accelerate"}


def decision(cp):
    fr = json.load(open(os.path.join(cp, "metadata.json")))["frames"]
    for k in (100, 101, 99):
        r = fr[k].get("reasoning")
        if r and os.path.isfile(os.path.join(cp, r)):
            dr = json.load(open(os.path.join(cp, r))).get("Driving")
            d = dr.get("Driving decision") if isinstance(dr, dict) else None
            if isinstance(d, dict) and d.get("Longitudinal") and d.get("Lateral"):
                return LON_MAP.get(d["Longitudinal"], d["Longitudinal"]), d["Lateral"]
    return None


def kin(split):
    ps = [np.load(f) for f in sorted(glob.glob(f"artifacts/labels148/{split}_[0-9]*.npz"))]
    return {n: f for p in ps for n, f in zip(p["names"], p["feat"])}


class Net(nn.Module):
    def __init__(self, nkin, nlon, nlat, d=256, vision=True):
        super().__init__()
        self.vision = vision
        self.proj = nn.Linear(768, d); self.pos = nn.Parameter(torch.zeros(1, 2 * 8 * 49, d))
        self.kin = nn.Sequential(nn.Linear(nkin, d), nn.GELU(), nn.Linear(d, d))
        self.q = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.lon = nn.Linear(d, nlon); self.lat = nn.Linear(d, nlat)

    def forward(self, v, k):
        toks = [self.q.expand(len(k), -1, -1), self.kin(k)[:, None]]
        if self.vision:
            toks.append(self.proj(v) + self.pos)
        h = self.enc(torch.cat(toks, 1))[:, 0]
        return self.lon(h), self.lat(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-root", default="feat/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.manual_seed(0); dev = "cuda"
    data = {}
    for split in ("train", "validation", "test"):
        K = kin(split); rows = []
        for n in sorted(K):
            fp = f"{a.feat_root}/{split}/{n}.npy"
            if not os.path.isfile(fp): continue
            y = decision(f"cache/{split}/{n}") if split != "test" else None
            if split == "train" and y is None: continue
            rows.append((n, fp, np.nan_to_num(K[n]).astype(np.float32), y))
        data[split] = rows; print(split, len(rows), flush=True)
    LON = sorted({r[3][0] for r in data["train"]}); LAT = sorted({r[3][1] for r in data["train"]})
    kin_mu = np.mean([r[2] for r in data["train"]], 0); kin_sd = np.std([r[2] for r in data["train"]], 0) + 1e-6

    cache = {}
    def tensors(rows):
        for r in rows:
            if r[1] not in cache: cache[r[1]] = np.load(r[1]).reshape(-1, 768).astype(np.float16)
        V = torch.from_numpy(np.stack([cache[r[1]] for r in rows]))
        Kt = torch.from_numpy(((np.stack([r[2] for r in rows]) - kin_mu) / kin_sd).astype(np.float32))
        return V, Kt

    def fit(rows):
        V, Kt = tensors(rows)
        yl = torch.tensor([LON.index(r[3][0]) for r in rows]); ya = torch.tensor([LAT.index(r[3][1]) for r in rows])
        m = Net(Kt.shape[1], len(LON), len(LAT), vision=not a.no_vision).to(dev)
        opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=0.05)
        steps = a.epochs * ((len(rows) + 63) // 64); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps)
        for ep in range(a.epochs):
            m.train(); perm = torch.randperm(len(rows))
            for i in range(0, len(rows), 64):
                b = perm[i:i + 64]
                ol, oa = m(V[b].to(dev).float(), Kt[b].to(dev))
                loss = F.cross_entropy(ol, yl[b].to(dev)) + F.cross_entropy(oa, ya[b].to(dev))
                opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        return m

    @torch.no_grad()
    def predict(m, rows):
        m.eval(); V, Kt = tensors(rows); out = []
        for i in range(0, len(rows), 256):
            ol, oa = m(V[i:i + 256].to(dev).float(), Kt[i:i + 256].to(dev))
            out.append(torch.cat([ol.softmax(-1), oa.softmax(-1)], -1).cpu().numpy())
        return np.concatenate(out)

    log_of = lambda n: "_".join(n.split("_")[:2])
    tr = data["train"]; logs = sorted({log_of(r[0]) for r in tr}); random.Random(0).shuffle(logs)
    fold = {l: i % 5 for i, l in enumerate(logs)}
    oof = np.zeros((len(tr), len(LON) + len(LAT)), np.float32)
    P = {s: [] for s in ("validation", "test")}
    for f in range(5):
        t0 = time.time()
        idx_tr = [i for i, r in enumerate(tr) if fold[log_of(r[0])] != f]; idx_te = [i for i, r in enumerate(tr) if fold[log_of(r[0])] == f]
        m = fit([tr[i] for i in idx_tr])
        oof[idx_te] = predict(m, [tr[i] for i in idx_te])
        for s in P: P[s].append(predict(m, data[s]))
        print(f"fold {f} done {time.time()-t0:.0f}s", flush=True)
    yl = np.array([LON.index(r[3][0]) for r in tr]); ya = np.array([LAT.index(r[3][1]) for r in tr])
    print(f"train OOF accuracy  lon {np.mean(oof[:, :len(LON)].argmax(1) == yl):.3f}  lat {np.mean(oof[:, len(LON):].argmax(1) == ya):.3f}"
          f"   (majority lon {np.bincount(yl).max()/len(yl):.3f}, lat {np.bincount(ya).max()/len(ya):.3f})")
    vy = [r[3] for r in data["validation"]]; pv = np.mean(P["validation"], 0)
    ok = [i for i, y in enumerate(vy) if y and y[0] in LON and y[1] in LAT]
    print(f"validation accuracy lon {np.mean([pv[i, :len(LON)].argmax() == LON.index(vy[i][0]) for i in ok]):.3f}  "
          f"lat {np.mean([pv[i, len(LON):].argmax() == LAT.index(vy[i][1]) for i in ok]):.3f}  (n={len(ok)})")
    for s, M in (("train", oof), ("validation", pv), ("test", np.mean(P["test"], 0))):
        rows = data[s]
        np.savez_compressed(f"artifacts/decpred{a.tag}_{s}.npz", names=np.array([r[0] for r in rows]), clip=M.astype(np.float32),
                            clr=np.zeros((len(rows), 148, 2), np.float32), clip_cols=np.array(LON + LAT))
    print("-> artifacts/decpred%s_{train,validation,test}.npz" % a.tag)


if __name__ == "__main__":
    main()
