"""2026-11-14: horizontal-flip test-time augmentation for the distance nets (they were trained with random flips; the
geometry vector is not flipped in training either). For each net dir: load model.pt, predict validation + test distance
questions on flipped crops -> <net>/pred_{validation,test}_flip.json. TTA = geometric mean of pred and pred_flip
(scored on the Mac by src/dist_tta_eval.py).
usage: python src/dist_tta.py runs_dist/ens_h16_s0 [...]
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dist_net import load, DS, Net


def main():
    vq = [it for it in load("box3d_q_validation") if it[2]["sub"] == "distance_from_2d_bbox"]
    tq = [it for it in load("box3d_q_test") if it[2]["sub"] == "distance_from_2d_bbox"]
    print(f"validation {len(vq)} test {len(tq)}", flush=True)
    for d in sys.argv[1:]:
        st = torch.load(os.path.join(d, "model.pt"), map_location="cpu", weights_only=False)
        net = Net(len(st["mu"]), st["model"]); net.load_state_dict(st["state"]); net = net.cuda().eval()
        for name, its in (("validation", vq), ("test", tq)):
            dl = torch.utils.data.DataLoader(DS(its, st["mu"], st["sd"], False), batch_size=128, num_workers=8)
            out = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for x, g, _, _ in dl: out.append(net(x.flip(-1).cuda(), g.cuda()).float().cpu())
            p = np.exp(torch.cat(out).numpy())
            json.dump({s["qid"]: float(v) for v, (_, _, s) in zip(p, its)}, open(os.path.join(d, f"pred_{name}_flip.json"), "w"))
        print(f"{d}: flip predictions saved", flush=True)
    print("TTA_DONE", flush=True)


if __name__ == "__main__":
    main()
