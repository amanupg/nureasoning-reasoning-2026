#!/bin/bash
# usage: a15ft_infer_latest.sh <model-half> <extra args for a15ft_infer.py...>
H=$1; shift
CK=$(ls -d runs_a15ft/$H/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
[ -z "$CK" ] && { echo "no checkpoint for $H"; exit 2; }
echo "checkpoint: $CK"
exec ext/a15ftenv/bin/python src/a15ft_infer.py --ckpt $CK "$@"
