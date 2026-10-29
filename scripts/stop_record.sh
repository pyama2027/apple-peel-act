#!/bin/bash
# Stop record_v2.sh first (so it does not go on to merge / signal RECORD_V2_DONE), then its recorder processes.
pkill -f 'bash scripts/record_v2.sh'
sleep 1
pkill -f 'python record_dataset.py'
sleep 3
echo "record procs left: $(pgrep -fc 'python record_dataset.py')  record_v2.sh left: $(pgrep -fc 'scripts/record_v2.sh')"
echo "pipeline waiting: $(pgrep -fc after_record_v2.sh)"
cat ~/apple_peel/outputs/record_v2.log
