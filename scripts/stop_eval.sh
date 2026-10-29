#!/bin/bash
# Stop in-training checkpoint evals (wrapper first so it does not copy partial results), drop their partial outputs.
cd ~/apple_peel
pkill -f 'bash scripts/eval_ckpt.sh'
sleep 1
pkill -f 'python -u eval_policy.py'
sleep 3
rm -rf outputs/eval/act_peel_v2_040000
echo "eval_ckpt: $(pgrep -fc 'bash scripts/eval_ckpt.sh')  eval_policy: $(pgrep -fc 'python -u eval_policy.py')"
echo "train: $(pgrep -fc lerobot-train)  eval_all_v2 waiting: $(pgrep -fc 'bash scripts/eval_all_v2.sh')"
ls outputs/eval/
