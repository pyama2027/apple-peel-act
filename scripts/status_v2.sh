#!/bin/bash
# Quick status of the v2 record / train / eval pipeline.
cd ~/apple_peel
echo "time $(date +%H:%M)  load $(cut -d' ' -f1-3 /proc/loadavg)"
echo "record procs: $(pgrep -fc 'python record_dataset.py')  train: $(pgrep -fc 'lerobot-train')  eval: $(pgrep -fc 'eval_policy.py')  pipeline: $(pgrep -fc after_record_v2.sh)"
tried=$(cat outputs/record_v2/shard*.log 2>/dev/null | grep -c '"seed"')
ok=$(cat outputs/record_v2/shard*.log 2>/dev/null | grep -c '"success": true')
echo "episodes tried $tried success $ok"
for f in outputs/record_v2/shard*.log; do
  echo "$(basename $f): $(grep -c '"seed"' $f) tried, last: $(tail -c 200 $f | tr '\n' ' ' | tail -c 160)"
done
grep -l -E 'Traceback' outputs/record_v2/shard*.log 2>/dev/null | sed 's/^/TRACEBACK: /'
tail -n 2 outputs/record_v2.log outputs/after_record_v2.log 2>/dev/null
grep -a 'step:' outputs/train_act_peel_v2.log 2>/dev/null | tail -1 | sed 's/.*step:/step:/'
