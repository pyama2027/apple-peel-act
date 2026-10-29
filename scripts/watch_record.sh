#!/bin/bash
# Print a line every 10 successful recording episodes (and on errors); exit when the recorders are gone.
cd ~/apple_peel/outputs
last=-1
while true; do
  n=$(cat record_v2/shard*.log 2>/dev/null | grep -c '"seed"')
  ok=$(cat record_v2/shard*.log 2>/dev/null | grep -c '"success": true')
  if [ $((ok / 10)) != "$last" ]; then echo "$(date +%H:%M) tried $n success $ok"; last=$((ok / 10)); fi
  grep -l Traceback record_v2/shard*.log 2>/dev/null | head -2 | sed 's/^/TRACEBACK in /'
  if [ "$(pgrep -fc 'python record_dataset.py')" = 0 ]; then echo "recorders exited (tried $n success $ok)"; tail -n 3 record_v2.log; break; fi
  sleep 30
done
