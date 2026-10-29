#!/bin/bash
# sheets.sh CKPT seed... -> contact sheets of the v2 eval videos into Downloads/act_eval_v2_20261028/CKPT/
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot
ck=$1; shift
cd ~/apple_peel/outputs/eval/act_peel_v2_$ck
for s in "$@"; do
  python ~/apple_peel/scripts/grid.py eval_seed$s.mp4 /mnt/c/Users/<WINDOWS_USER>/Downloads/act_eval_v2_20261028/$ck/sheet_$s.png 4 \
    2 4 6 8 10 12 15 20 30 60 120 240
done
