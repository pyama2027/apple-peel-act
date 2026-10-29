"""Day1 check: IK to top-down poses above the apple and the needle, render overhead/wrist views at each."""
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from ik import ARM_JOINTS, solve_ik, top_down_targets

ROOT = Path(__file__).resolve().parents[1]
m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/models/so101/apple_scene.xml"))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
apple = d.xpos[m.body("apple").id].copy()
needle_tip = d.site_xpos[m.site("needle_tip").id].copy()
qadr = [m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS]

poses = {
    "above_apple_10cm": apple + [0, 0, 0.10],
    "at_apple_center": apple,
    "above_needle_12cm": needle_tip + [0, 0, 0.12],
}
r = mujoco.Renderer(m, 224, 224)
rows = []
q = np.zeros(len(ARM_JOINTS))
for name, tip in poses.items():
    q, err = solve_ik(m, top_down_targets(tip), q)
    d.qpos[qadr] = q
    d.qpos[m.jnt_qposadr[m.joint("gripper").id]] = 1.2  # open
    mujoco.mj_forward(m, d)
    tip_now = d.site_xpos[m.site("gripperframe").id]
    print(f"{name:18s} err={err * 1000:.1f} mm  tip={np.round(tip_now, 3)}  q[deg]={np.round(np.degrees(q), 1)}")
    row = []
    for cam in ("overhead", "wrist"):
        r.update_scene(d, camera=cam)
        row.append(r.render())
    rows.append(np.concatenate(row, axis=1))
r.close()
Image.fromarray(np.concatenate(rows, axis=0)).save(ROOT / "outputs/day1_scene/ik_views.png")
print("saved outputs/day1_scene/ik_views.png")
