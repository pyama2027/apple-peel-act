"""Does an apple of radius r fit between the open jaws? Place the tip at the apple center (top-down IK),
open the gripper fully, and measure penetration between the apple and robot geoms for several radii and grasp depths."""
from pathlib import Path

import mujoco
import numpy as np

from ik import ARM_JOINTS, solve_ik, top_down_targets

ROOT = Path(__file__).resolve().parents[1]
m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/models/so101/apple_scene.xml"))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
apple_g = m.geom("apple").id
apple_b = m.body("apple").id
qadr = [m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS]
grip = m.jnt_qposadr[m.joint("gripper").id]
robot_bodies = {m.body(n).id for n in ("gripper", "moving_jaw_so101_v1", "wrist", "lower_arm")}

for r in (0.030, 0.035, 0.040):
    m.geom_size[apple_g, 0] = r
    m.body_pos[apple_b, 2] = r
    for depth in (0.0, 0.01, 0.02):  # how far the tip goes below the apple center
        mujoco.mj_resetData(m, d)  # also re-reads apple start pos from body_pos
        mujoco.mj_forward(m, d)
        center = d.xpos[apple_b].copy()
        q, err = solve_ik(m, top_down_targets(center - [0, 0, depth]), np.zeros(5))
        d.qpos[qadr] = q
        for gopen in (1.2, 1.745):
            d.qpos[grip] = gopen
            mujoco.mj_forward(m, d)
            worst = 0.0
            for c in d.contact[:d.ncon]:
                pair = {c.geom1, c.geom2}
                if apple_g in pair:
                    other = (pair - {apple_g}).pop()
                    if m.geom_bodyid[other] in robot_bodies:
                        worst = min(worst, c.dist)
            print(f"r={r * 100:.1f}cm depth={depth * 100:.0f}cm grip={gopen:.2f}rad ik_err={err * 1000:.1f}mm "
                  f"max_penetration={-worst * 1000:.1f}mm")
