#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 6:00:00
#SBATCH -o /path/to/cluster/scratch/nureasoning/logs/push_to_vm_%j.out
# 2026-11-17: push the dense LoRA continuation inputs to a rented VM (Nebius, Aman's account) over SSH with the
# dedicated key ~/.ssh/pod_ed25519 (public half registered on the VM). From a compute node, not the login node.
# Sends: code + v1 dense artifacts + LoRA state (one small tar), the model parts, image chunks for img_vlm/train and
# img_dense_A/B (v1 data = exactly what the Insomnia run trains on). NP parallel scp streams; md5 list alongside.
# usage: VM=nr@<ip> sbatch cluster/push_to_vm.sbatch
source /etc/profile
P=/path/to/cluster/scratch/nureasoning; cd $P
SSHO="-i $HOME/.ssh/pod_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/known_hosts_pod -o ServerAliveInterval=30"
D=/dev/shm/$USER_push_$SLURM_JOB_ID; mkdir -p $D; trap "rm -rf $D" EXIT
tar cf $D/code_artifacts.tar src cluster pod artifacts/dense_index_train.json artifacts/dense_q_train.jsonl \
  artifacts/dense_members_train_A.json artifacts/dense_members_train_B.json artifacts/clipfeat_train.json \
  runs_vlm/qd4b_dense_lr1e4/last/state.pt runs_vlm/qd4b_dense_lr1e4/args.json
( cd $D && md5sum code_artifacts.tar > code.md5 )
ssh $SSHO $VM "mkdir -p ~/w/packs ~/w/packs2" || { echo "SSH_FAIL"; exit 1; }
scp $SSHO $D/code_artifacts.tar $D/code.md5 $VM:w/ && echo "[push] code_artifacts.tar sent"
scp $SSHO packs/models_qwen_drive_vlm.files packs/MD5SUMS $VM:w/packs/
ls packs2/{img_vlm_train,img_dense_A_train,img_dense_B_train}.c*.files | xargs -I{} scp $SSHO {} $VM:w/packs2/
grep -E "img_vlm_train|img_dense_A_train|img_dense_B_train" packs2/MD5SUMS > $D/MD5SUMS; scp $SSHO $D/MD5SUMS $VM:w/packs2/
t0=$(date +%s)
( ls packs/models_qwen_drive_vlm.tar.part-*; ls packs2/{img_vlm_train,img_dense_A_train,img_dense_B_train}.c*.tar.part-* ) \
  | xargs -P ${NP:-8} -I{} sh -c "scp $SSHO {} $VM:w/{} && echo \"[push] {} \$(( \$(date +%s) - $t0 )) s\""
echo "[push] all parts sent in $(( $(date +%s) - t0 )) s"; echo PUSH_DONE
