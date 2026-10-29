"""Which robot geoms does a 3 cm apple touch when placed at the gripperframe with jaws open? Also list jaw collision geoms."""
from pathlib import Path

import mujoco
import numpy as np

from ik import ARM_JOINTS, solve_ik, top_down_targets

ROOT = Path(__file__).resolve().parents[1]
m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/models/so101/apple_scene.xml"))
d = mujoco.MjData(m)
name = lambda t, i: mujoco.mj_id2name(m, t, i)

for b in ("gripper", "moving_jaw_so101_v1"):
    for g in np.where(m.geom_bodyid == m.body(b).id)[0]:
        if m.geom_contype[g] or m.geom_conaffinity[g]:
            mesh = name(mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else "-"
            print(f"collision geom #{g} body={b} mesh={mesh}")

apple_g, apple_b = m.geom("apple").id, m.body("apple").id
m.geom_size[apple_g, 0] = 0.03
m.body_pos[apple_b, 2] = 0.03
mujoco.mj_resetData(m, d)
mujoco.mj_forward(m, d)
q, _ = solve_ik(m, top_down_targets(d.xpos[apple_b].copy()), np.zeros(5))
d.qpos[[m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS]] = q
d.qpos[m.jnt_qposadr[m.joint("gripper").id]] = 1.745
mujoco.mj_forward(m, d)
for c in d.contact[:d.ncon]:
    if apple_g in (c.geom1, c.geom2):
        o = c.geom2 if c.geom1 == apple_g else c.geom1
        mesh = name(mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[o]) if m.geom_type[o] == mujoco.mjtGeom.mjGEOM_MESH else "-"
        print(f"apple touches geom #{o} body={name(mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[o])} mesh={mesh} dist={c.dist * 1000:.1f}mm")

# where is gripperframe relative to the two jaws? (world positions)
print("gripperframe", np.round(d.site_xpos[m.site('gripperframe').id], 4))
print("moving jaw body", np.round(d.xpos[m.body('moving_jaw_so101_v1').id], 4))
