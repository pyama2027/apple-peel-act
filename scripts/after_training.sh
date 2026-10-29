#!/bin/bash
# Wait for the ACT training run to finish, then evaluate the final checkpoint on 6 unseen apple placements (with
# review videos, 6 processes in parallel) and copy videos + results to the Windows Downloads folder.
# Writes outputs/eval/act_peel/ALL_DONE when finished. Usage: setsid nohup bash after_training.sh &
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot
cd "$(dirname "$0")/../sim"
OUT=../outputs/eval/act_peel
DL=/mnt/c/Users/<WINDOWS_USER>/Downloads/act_eval_20261027
mkdir -p $OUT
while pgrep -f lerobot-train > /dev/null; do sleep 60; done
echo "training finished $(date)" > $OUT/status.txt
CKPT=../outputs/train/act_peel/checkpoints/last/pretrained_model
for s in 5000 5001 5002 5003 5004 5005; do
  python -u eval_policy.py --policy $CKPT --seeds $s --video --out $OUT > $OUT/log_$s.txt 2>&1 &
done
wait
echo "eval finished $(date)" >> $OUT/status.txt
mkdir -p $DL
cp $OUT/eval_seed*.mp4 $OUT/eval_seed*.json $DL/
grep -a "step:" ../outputs/train_act_peel.log | tail -1 | sed 's/.*step:/step:/' > $DL/training_last_step.txt
touch $OUT/ALL_DONE
