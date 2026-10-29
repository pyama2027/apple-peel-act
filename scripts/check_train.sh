#!/bin/bash
cd ~/apple_peel/outputs
date +%H:%M
echo "pipeline procs: $(pgrep -fc after_record_v2.sh)  train procs: $(pgrep -fc lerobot-train)"
echo "--- after_record_v2.log"; cat after_record_v2.log
echo "--- train log"; ls -la train_act_peel_v2.log 2>&1; tail -c 400 train_act_peel_v2.log 2>/dev/null; echo
grep -c RECORD_V2_DONE record_v2.log
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheade
