#!/bin/bash
# sheet_one.sh EVALDIR SEED OUT.png [times...] -> contact sheet of one eval video (default times 2..60 s)
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot
d=$1; s=$2; out=$3; shift 3
t=${@:-"2 4 6 8 10 12 14 16 20 25 30 40"}
python ~/apple_peel/scripts/grid.py ~/apple_peel/outputs/eval/$d/eval_seed$s.mp4 $out 4 $t
