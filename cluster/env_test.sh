#!/bin/bash
P=/path/to/cluster/scratch/nureasoning; cd $P
source cluster/stage_env.sh
$PY - <<'PY'
import sys, torch, transformers, peft
print("[envtest] ok", torch.__version__, transformers.__version__, peft.__version__, sys.executable, torch.__file__)
PY
