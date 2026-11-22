# Copy the Python env (and optional model dirs) to node-local disk and run from there (2026-11-06: shared GPFS
# at ~20 MB/s left every job stuck >40 min page-faulting torch's .so files; big sequential copies are faster).
# usage (after `source cluster/stage_img.sh ...` if used):  STAGE_MODELS="models/qwen_drive_vlm" source cluster/stage_env.sh
# exports PY (python on /local) and LM (local root for the copied model dirs: $LM/models/...).
LE=/local/$USER_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}_env
mkdir -p $LE; t0=$(date +%s)
trap "rm -rf ${STAGE:-/nonexistent_} $LE" EXIT
if [ -f $P/env_snapshot.tar ] && [ $P/env_snapshot.tar -nt $P/env/bin/python3.12 ]; then
  tar -C $LE -xf $P/env_snapshot.tar            # one big sequential read (built by cluster/env_snapshot.sh)
else
  tar -C $P -cf - env | tar -C $LE -xf -
fi
for m in ${STAGE_MODELS:-}; do mkdir -p $LE/$(dirname $m)
  # 2026-11-08: a $m.tar newer than every file in $m is one sequential read (img_crops: 2.4 h of small files -> tar)
  if [ -f $P/$m.tar ] && [ -z "$(find $P/$m -newer $P/$m.tar -print -quit)" ]; then tar -C $LE/$(dirname $m) -xf $P/$m.tar
  else tar -C $P -cf - $m | tar -C $LE -xf -; fi
done
export PY=$LE/env/bin/python LM=$LE
$PY -c "import sys; print('[env] python', sys.prefix)" && echo "[env] env + models (${STAGE_MODELS:-none}) -> $LE in $(( $(date +%s) - t0 )) s, $(du -sh $LE | cut -f1)"
