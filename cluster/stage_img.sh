# Stage raw frames for THIS job onto node-local disk instead of keeping them on the shared fileset.
# usage (inside an sbatch script, after cd $P and HF_HOME are set):
#   source cluster/stage_img.sh <splits, e.g. test or train,validation>
# Fetches the same full-resolution members as the old img/ folder (src/image_members.json) straight from the
# HF dataset tars (src/img_ingest2.py; saved tar indexes in img_index/ make it fast), into
# /local/$USER_<jobid>/img, exports IMG_ROOT there (read by vlm_data, fm_track, box3d_embed, dist_depth),
# and deletes the folder when the job exits.
STAGE=/local/$USER_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p $STAGE/img
trap "rm -rf $STAGE" EXIT
t0=$(date +%s)
SLURM_PROCID=0 SLURM_NTASKS=1 env/bin/python -u src/img_ingest2.py --splits $1 --out $STAGE/img \
    --index-from img_index/img:img_index/img_a15:img_index/img_full --clips ${STAGE_CLIPS:-8} --K ${STAGE_K:-24} 2>&1 | grep --line-buffered -E "DONE|FAIL|^\[img2\]"
export IMG_ROOT=$STAGE/img
echo "[stage] $1 -> $IMG_ROOT in $(( $(date +%s) - t0 )) s, $(du -sh $STAGE/img | cut -f1)"
