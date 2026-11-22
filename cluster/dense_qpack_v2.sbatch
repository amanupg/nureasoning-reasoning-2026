#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=24G
#SBATCH -t 2:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/dense_qpack_v2_%j.out
# 2026-11-16: v2 question pack (key + passes A/B/C) for the rented-GPU run. Does not touch the v1 files the running jobs use.
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
DENSE_INDEX=artifacts/dense_index_train_v2.json DENSE_QPACK=artifacts/dense_q_train_v2.jsonl env/bin/python -u src/dense_qpack.py
wc -l artifacts/dense_q_train_v2.jsonl artifacts/dense_q_train.jsonl; md5sum artifacts/dense_q_train.jsonl
echo QPACK_DONE
