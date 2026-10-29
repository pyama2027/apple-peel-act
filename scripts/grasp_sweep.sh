#!/bin/bash
# Grasp capacity sweep: apple radius x gripper opening (sim/test_grasp.py). noslip is now set in the scene.
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot
cd "$(dirname "$0")/../sim"
export MUJOCO_GL=egl
for o in 1.0 1.3 1.5; do
  echo "### open=$o"
  python -u test_grasp.py --radii 0.030 0.035 0.040 --open $o 2>&1 | grep -E "^r="
done
