#!/bin/bash
# Expert success rate over random apple placements (metrics only). Usage: expert_sweep.sh N_SEEDS
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot
cd "$(dirname "$0")/../sim"
N=${1:-20}
OUT=../outputs/day2_sweep
mkdir -p $OUT
for s in $(seq 100 $((99 + N))); do
  python -u run_demo.py --seed $s --no-video --out $OUT 2>/dev/null | grep '"seed"'
done
