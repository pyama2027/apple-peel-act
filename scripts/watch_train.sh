#!/bin/bash
# Print the first training log line of every 10K steps (and errors); exit when training is over.
cd ~/apple_peel/outputs
seen=""
while true; do
  for k in $(grep -a -o 'step:[0-9]*0K' train_act_peel_v2.log 2>/dev/null | sort -u); do
    case " $seen " in *" $k "*) ;; *)
      echo "$(date +%H:%M) $(grep -a "$k " train_act_peel_v2.log | head -1 | sed 's/.*step:/step:/' | cut -c1-110)"
      seen="$seen $k";;
    esac
  done
  grep -a -m1 -E 'Traceback|out of memory' train_act_peel_v2.log 2>/dev/null | sed 's/^/ERROR: /'
  if ! pgrep -f lerobot-train > /dev/null; then echo "training not running $(date +%H:%M)"; tail -n 2 after_record_v2.log; break; fi
  sleep 60
done
