"""Design finger extensions (option B: keep the 8 cm apple).

Extensions are defined in the gripper frame at gripper q=0 (fingers parallel, approach = -z, jaws face along x),
then the moving-jaw one is converted into the moving_jaw body frame so it rotates with the jaw.
Prints XML snippets to paste into so101_new_calib_camera.xml.
"""
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXT_Z = (-0.095, -0.145)          # extension span along the approach axis (original finger tips end ~-0.104)
HALF_X, HALF_Y = 0.005, 0.008     # finger thickness (x, towards the other jaw) and width (y)
FIXED_INNER_X, MOVING_INNER_X = -0.010, 0.010  # inner faces at q=0 (sim/probe_jaw_geometry.py)

m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/models/so101/apple_scene.xml"))
d = mujoco.MjData(m)
d.qpos[m.jnt_qposadr[m.joint("gripper").id]] = 0.0
mujoco.mj_forward(m, d)
g, j = m.body("gripper").id, m.body("moving_jaw_so101_v1").id
Rg, pg = d.xmat[g].reshape(3, 3), d.xpos[g]
Rj, pj = d.xmat[j].reshape(3, 3), d.xpos[j]

zc, zh = (EXT_Z[0] + EXT_Z[1]) / 2, (EXT_Z[0] - EXT_Z[1]) / 2
fixed_c = np.array([FIXED_INNER_X - HALF_X, 0, zc])
moving_c = np.array([MOVING_INNER_X + HALF_X, 0, zc])

# moving extension: gripper-frame pose -> moving-jaw-frame pose
pos_j = Rj.T @ (pg + Rg @ moving_c - pj)
R_j = Rj.T @ Rg
quat = np.zeros(4)
mujoco.mju_mat2Quat(quat, R_j.ravel())

f = lambda v: " ".join(f"{x:.5f}" for x in v)
size = f"{HALF_X} {HALF_Y} {zh:.4f}"
print("<!-- in body gripper -->")
print(f'<geom name="fixed_finger_ext" type="box" class="visual" pos="{f(fixed_c)}" size="{size}" material="finger_ext"/>')
print(f'<geom name="fixed_finger_ext_col" type="box" class="collision" pos="{f(fixed_c)}" size="{size}" friction="1.5 0.02 0.001" condim="4"/>')
print("<!-- in body moving_jaw_so101_v1 -->")
print(f'<geom name="moving_finger_ext" type="box" class="visual" pos="{f(pos_j)}" quat="{f(quat)}" size="{size}" material="finger_ext"/>')
print(f'<geom name="moving_finger_ext_col" type="box" class="collision" pos="{f(pos_j)}" quat="{f(quat)}" size="{size}" friction="1.5 0.02 0.001" condim="4"/>')
