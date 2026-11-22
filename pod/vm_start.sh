#!/bin/bash
cd ~/w; . ~/env/bin/activate
ls -la runs_vlm/qd4b_dense_lr1e4/last/state.pt; ls artifacts
python -c "import torch; s=torch.load('runs_vlm/qd4b_dense_lr1e4/last/state.pt', map_location='cpu', weights_only=False); print('state step', s['step'], 'tensors', len(s['adapter']))"
touch UNPACK_OK; sudo systemctl start nr-lora.service; sleep 5; systemctl is-active nr-lora.service; cat boot.log
