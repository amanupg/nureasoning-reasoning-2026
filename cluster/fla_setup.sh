#!/bin/bash
# Install flash-linear-attention (Triton kernels for Qwen3.5 GatedDeltaNet) into env and benchmark it.
P=/path/to/cluster/scratch/nureasoning; cd $P
export PYTHONNOUSERSITE=1 TMPDIR=/dev/shm/$USER_fla_$$; mkdir -p $TMPDIR; trap "rm -rf $TMPDIR" EXIT
env/bin/python -c "import triton, torch; print('triton', triton.__version__, 'torch', torch.__version__)"
env/bin/pip install -q --no-deps flash-linear-attention fla-core 2>&1 | tail -3
env/bin/pip show fla-core flash-linear-attention 2>/dev/null | grep -E "^(Name|Version)"
env/bin/python - <<'PY'
import torch, time
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
B,T,H,K=1,4096,16,128
q=torch.randn(B,T,H,K,device="cuda",dtype=torch.bfloat16); k=torch.nn.functional.normalize(torch.randn_like(q),dim=-1); v=torch.randn_like(q)
g=-torch.rand(B,T,H,device="cuda",dtype=torch.float32); b=torch.rand(B,T,H,device="cuda",dtype=torch.bfloat16)
for i in range(3):
    torch.cuda.synchronize(); t=time.time(); o,_=chunk_gated_delta_rule(q,k,v,g,b); torch.cuda.synchronize(); print("fla ok", tuple(o.shape), round(time.time()-t,4), "s")
PY
