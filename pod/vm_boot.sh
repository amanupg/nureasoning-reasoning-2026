#!/bin/bash
# runs at every boot (systemd nr-lora.service) and on manual start: re-arm the 13:30 UTC hard stop, then resume
# the LoRA run if unpacking is complete and the final step does not exist yet.
STOP=$(date -u -d "2026-11-17 13:30" +%s); now=$(date -u +%s)
if [ $now -ge $STOP ]; then shutdown -h now; exit 0; fi
shutdown -h 13:30
cd /home/nr/w
[ -f UNPACK_OK ] || { echo "$(date -u) boot: unpack not done, not training" >> boot.log; exit 0; }
[ -d runs_vlm/qd4b_dense_lr1e4/step_3500 ] && { touch TRAIN_DONE; echo "$(date -u) boot: already done" >> boot.log; exit 0; }
echo "$(date -u) boot: starting training" >> boot.log
exec sudo -u nr -H bash -c 'bash /home/nr/w/vm_lora_resume.sh; [ -d /home/nr/w/runs_vlm/qd4b_dense_lr1e4/step_3500 ] && touch /home/nr/w/TRAIN_DONE'
