#!/bin/bash
# eval_ckpt.sh CKPT -> wait until that v2 checkpoint is saved, evaluate it on seeds 5000-5005 with videos (6 parallel,
# software GL) while training goes on, copy to Downloads/act_eval_v2_20261028/CKPT/ and refresh 結果一覧.md.
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot
export LIBGL_ALWAYS_SOFTWARE=1
cd ~/apple_peel
ck=$1
P=outputs/train/act_peel_v2/checkpoints/$ck/pretrained_model
until [ -e $P/model.safetensors ]; do sleep 30; done
sleep 30   # let the checkpoint finish writing
echo "$ck saved $(date +%H:%M)"
OUT=outputs/eval/act_peel_v2_$ck
mkdir -p $OUT
cd sim
for s in 5000 5001 5002 5003 5004 5005; do
  python -u eval_policy.py --policy ../$P --seeds $s --max-frames 7500 --video --out ../$OUT > ../$OUT/log_$s.txt 2>&1 &
done
wait
DL=/mnt/c/Users/<WINDOWS_USER>/Downloads/act_eval_v2_20261028
mkdir -p $DL/$ck
cp ../$OUT/eval_seed* $DL/$ck/
python ../scripts/eval_table.py > $DL/結果一覧.md
echo "EVAL_$ck DONE $(date +%H:%M)"
