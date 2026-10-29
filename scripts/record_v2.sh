#!/bin/bash
# Record the v2 peel dataset (2026-10-28 diversity: fence, peeler moved, apple / peeler drops, skewer misses,
# needle / peeler / start-posture jitter) with 10 processes, then merge. v1 data was moved to data/_v1/.
# Software GL: the default (WSLg GPU) path serialises the 10 renderers (0.78 s/frame vs 0.18 s/frame).
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot
export LIBGL_ALWAYS_SOFTWARE=1
cd ~/apple_peel
mkdir -p outputs/record_v2
cd sim
N=10
for ((i=0; i<N; i++)); do
  python record_dataset.py --task peel --episodes 5 --max-tries 12 --seed0 4000 --shard $i --num-shards $N --overwrite \
    > ../outputs/record_v2/shard$i.log 2>&1 &
done
wait
python merge_shards.py --task peel > ../outputs/record_v2/merge.log 2>&1
tail -n 2 ../outputs/record_v2/merge.log
echo RECORD_V2_DONE
