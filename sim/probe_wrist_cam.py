"""Compute a wrist camera pose (in the gripper body frame) that sits at the wrist_camera mesh and looks at the jaw tip."""
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/models/so101/so101_new_calib_camera.xml"))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

g = m.body("gripper").id
R_g, p_g = d.xmat[g].reshape(3, 3), d.xpos[g]
to_local = lambda p: R_g.T @ (p - p_g)

cam = to_local(d.xpos[m.body("wrist_camera").id])
tip = to_local(d.site_xpos[m.site("gripperframe").id])
fwd = tip - cam
fwd /= np.linalg.norm(fwd)
# mesh of jaws spans the gripper's local x axis; use it to fix the roll
x_ref = np.array([1.0, 0, 0])
xaxis = x_ref - fwd * (fwd @ x_ref)
xaxis /= np.linalg.norm(xaxis)
yaxis = np.cross(-fwd, xaxis)  # MuJoCo camera looks along -z: z = -fwd, x × y = z
print("cam pos (gripper frame):", np.round(cam, 4))
print("tip pos (gripper frame):", np.round(tip, 4))
print("distance to tip [m]:", round(float(np.linalg.norm(tip - cam)), 4))
print(f'xyaxes="{" ".join(f"{v:.4f}" for v in np.r_[xaxis, yaxis])}"')
