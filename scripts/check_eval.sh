#!/bin/bash
# check_eval.sh CKPT -> progress of the v2 eval of that checkpoint
d=~/apple_peel/outputs/eval/act_peel_v2_$1
date +%H:%M
echo "eval procs: $(pgrep -fc 'python -u eval_policy.py')  train: $(pgrep -fc lerobot-train)"
cat ~/apple_peel/outputs/eval_$1.log 2>/dev/null
for s in 5000 5001 5002 5003 5004 5005; do
  echo "seed $s: mp4 $(stat -c %s $d/eval_seed$s.mp4 2>/dev/null) bytes, json $( [ -e $d/eval_seed$s.json ] && echo yes || echo no)"
done
ls $d/*.json 2>/dev/null | xargs -r cat | grep -E '"seed"|skewered|peeled"|success_frame'
grep -l Traceback $d/log_*.txt 2>/dev/null
