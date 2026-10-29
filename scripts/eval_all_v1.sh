#!/bin/bash
# Evaluate every v1 ACT checkpoint (020000..100000) in the V1 simulator (sim_v1: peeler at 0.20,-0.15, no fence, no
# scene jitter = the sim the v1 dataset was recorded in) on seeds 5000-5005 with videos, then fill in the v2 rollouts
# (V2 sim, sim/) that died last night. 6 in parallel (12 ran out of memory), software GL.
# Results -> Downloads/act_eval_v1_20261029/<ckpt>/ + 結果一覧.md, and Downloads/act_eval_v2_20261028/ refreshed.
# Usage: setsid nohup bash scripts/eval_all_v1.sh > outputs/eval_all_v1.log 2>&1 &
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot
export LIBGL_ALWAYS_SOFTWARE=1
cd ~/apple_peel
NPAR=6
run() {  # run SIMDIR POLICY OUT SEED
  while [ "$(jobs -rp | wc -l)" -ge $NPAR ]; do sleep 20; done
  mkdir -p $3
  [ -e $3/eval_seed$4.json ] && return
  (cd $1 && python -u eval_policy.py --policy ../$2 --seeds $4 --max-frames 7500 --video --out ../$3 > ../$3/log_$4.txt 2>&1) &
}
for ck in 020000 040000 060000 080000 100000; do
  for s in 5000 5001 5002 5003 5004 5005; do
    run sim_v1 outputs/train/act_peel/checkpoints/$ck/pretrained_model outputs/eval/act_peel_v1sim_$ck $s
  done
done
for ck in 060000 080000 100000; do
  for s in 5000 5001 5002 5003 5004 5005; do
    run sim outputs/train/act_peel_v2/checkpoints/$ck/pretrained_model outputs/eval/act_peel_v2_$ck $s
  done
done
wait
for v in v1:act_eval_v1_20261029:act_peel_v1sim_ v2:act_eval_v2_20261028:act_peel_v2_; do
  IFS=: read ver dir pre <<< "$v"
  DL=/mnt/c/Users/<WINDOWS_USER>/Downloads/$dir
  for ck in 020000 040000 060000 080000 100000; do
    mkdir -p $DL/$ck && cp outputs/eval/$pre$ck/eval_seed* $DL/$ck/
  done
  python scripts/eval_table.py $ver > $DL/結果一覧.md
  cat $DL/結果一覧.md
done
echo "EVAL_ALL_V1_DONE $(date)"
