"""Pool the samples of several Alpamayo-format outputs (e.g. the two out-of-fold fine-tuned
half-models) into one per-clip npz, so mean_global() gives the mean of the models' mean paths.
usage: python src/merge_a15_samples.py OUT_DIR SRC_DIR [SRC_DIR ...]"""
import os, sys, numpy as np
out, srcs = sys.argv[1], sys.argv[2:]
os.makedirs(out, exist_ok=True)
names = sorted(set.intersection(*[{f for f in os.listdir(s) if f.endswith(".npz")} for s in srcs]))
for f in names:
    Z = [np.load(os.path.join(s, f)) for s in srcs]
    assert all(np.allclose(z["pose"], Z[0]["pose"]) for z in Z), f"pose mismatch {f}"
    assert len({z["xyz"].shape[0] for z in Z}) == 1, f"unequal sample counts {f}"   # equal weights per model
    np.savez_compressed(os.path.join(out, f), xyz=np.concatenate([z["xyz"] for z in Z]),
                        rot=np.concatenate([z["rot"] for z in Z]), pose=Z[0]["pose"], nav=Z[0]["nav"],
                        cot=np.concatenate([np.atleast_1d(z["cot"]) for z in Z]))
print(out, len(names), "clips from", srcs)
