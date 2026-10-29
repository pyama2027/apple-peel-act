#!/bin/bash
# Wait for the v2 ACT training to finish, then evaluate every checkpoint not evaluated yet (100000 first, then 060000,
# 080000, 040000, 020000) on the same 6 unseen seeds with videos, 12 rollouts in parallel (software GL), write a
# success table and copy everything to Downloads/act_eval_v2_20261028/<checkpoint>/.
# Checkpoints already done by eval_ckpt.sh during training (6 result JSONs) are skipped.
# Usage: setsid nohup bash scripts/eval_all_v2.sh > outputs/eval_all_v2.log 2>&1 &
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot
export LIBGL_ALWAYS_SOFTWARE=1
cd ~/apple_peel
CK=outputs/train/act_peel_v2/checkpoints
until [ -e $CK/100000/pretrained_model ] && ! pgrep -f lerobot-train > /dev/null; do sleep 60; done
echo "training finished $(date)"
grep -a "step:" outputs/train_act_peel_v2.log | tail -1 | sed 's/.*step:/step:/'
while pgrep -f 'bash scripts/eval_ckpt.sh' > /dev/null; do sleep 30; done   # an in-training eval still running
DL=/mnt/c/Users/<WINDOWS_USER>/Downloads/act_eval_v2_20261028
cd sim
todo=""
for ck in 100000 060000 080000 040000 020000; do
  OUT=../outputs/eval/act_peel_v2_$ck
  if [ "$(ls $OUT/eval_seed*.json 2>/dev/null | wc -l)" = 6 ]; then echo "skip $ck (done)"; continue; fi
  todo="$todo $ck"
  mkdir -p $OUT
  for s in 5000 5001 5002 5003 5004 5005; do
    while [ "$(jobs -rp | wc -l)" -ge 12 ]; do sleep 20; done
    python -u eval_policy.py --policy ../$CK/$ck/pretrained_model --seeds $s \
      --max-frames 7500 --video --out $OUT > $OUT/log_$s.txt 2>&1 &
  done
done
wait
for ck in $todo; do
  mkdir -p $DL/$ck
  cp ../outputs/eval/act_peel_v2_$ck/eval_seed* $DL/$ck/
done
python ../scripts/eval_table.py > $DL/結果一覧.md
cat $DL/結果一覧.md
echo "EVAL_ALL_DONE $(date)"
