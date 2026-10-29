#!/bin/bash
# Watch every eval result (v1 in V1 sim, v2 in V2 sim); for each successful rollout make a x2 final video
# (user 2026-10-29: "1本でも成功した場合は、2倍速で最終動画") cut 5 s after success, into Downloads/最終動画/.
# Usage: setsid nohup bash scripts/watch_success.sh > outputs/watch_success.log 2>&1 &
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot
cd ~/apple_peel
OUT=/mnt/c/Users/<WINDOWS_USER>/Downloads/最終動画
while true; do
  for j in outputs/eval/act_peel_v1sim_*/eval_seed*.json outputs/eval/act_peel_v2_*/eval_seed*.json; do
    [ -e "$j" ] || continue
    sf=$(python -c "import json,sys; v=json.load(open(sys.argv[1]))['success_frame']; print('' if v is None else v)" $j)
    [ -z "$sf" ] && continue
    d=$(basename $(dirname $j)); s=$(basename $j .json); dst=$OUT/final_${d}_${s}_x2.mp4
    [ -e "$dst" ] && continue
    mkdir -p $OUT
    (cd sim && python make_final_video.py ../${j%.json}.mp4 $dst --speed 2 --until-sec $(( sf / 30 + 5 ))) \
      && echo "FINAL $dst $(date +%H:%M)"
  done
  grep -q EVAL_ALL_V1_DONE outputs/eval_all_v1.log && { echo "watch end $(date)"; break; }
  sleep 120
done
