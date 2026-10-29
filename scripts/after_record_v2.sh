#!/bin/bash
# Wait for record_v2.sh (merged dataset), train ACT on it (same settings as v1: batch 8, 100k steps), then evaluate
# the final checkpoint on 6 unseen seeds with videos and copy them to Downloads/act_eval_v2_20261028.
# Usage: setsid nohup bash scripts/after_record_v2.sh > outputs/after_record_v2.log 2>&1 &
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot
cd ~/apple_peel
until grep -q RECORD_V2_DONE outputs/record_v2.log 2>/dev/null; do sleep 60; done
echo "record done $(date)"
lerobot-train --dataset.repo_id=local/so101_peel_sim --dataset.root=$HOME/apple_peel/data/so101_peel_sim \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=8 --steps=100000 --save_freq=20000 --num_workers=8 \
  --output_dir=outputs/train/act_peel_v2 --job_name=act_peel_v2 --wandb.enable=false \
  > outputs/train_act_peel_v2.log 2>&1
echo "train done $(date)"
cd sim
OUT=../outputs/eval/act_peel_v2
DL=/mnt/c/Users/<WINDOWS_USER>/Downloads/act_eval_v2_20261028
mkdir -p $OUT $DL
CKPT=../outputs/train/act_peel_v2/checkpoints/last/pretrained_model
for s in 5000 5001 5002 5003 5004 5005; do
  python -u eval_policy.py --policy $CKPT --seeds $s --max-frames 7500 --video --out $OUT > $OUT/log_$s.txt 2>&1 &
done
wait
cp $OUT/eval_seed*.mp4 $OUT/eval_seed*.json $DL/
grep -a "step:" ../outputs/train_act_peel_v2.log | tail -1 | sed 's/.*step:/step:/' > $DL/training_last_step.txt
echo "eval done $(date)"
touch $OUT/ALL_DONE
