"""Measure the finger parts of the fixed and moving jaw meshes in the gripper body frame (approach axis = -z)."""
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/models/so101/apple_scene.xml"))
d = mujoco.MjData(m)
grip_q = m.jnt_qposadr[m.joint("gripper").id]
g = m.body("gripper").id


def verts_in_gripper(geom_id):
    mid = m.geom_dataid[geom_id]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    w = v @ d.geom_xmat[geom_id].reshape(3, 3).T + d.geom_xpos[geom_id]
    return (w - d.xpos[g]) @ d.xmat[g].reshape(3, 3)


fixed, moving = 29, 35  # collision geoms found by probe_gripper_contacts.py (wrist_roll_follower / moving_jaw meshes)
for gq in (-0.17, 0.0, 0.8, 1.745):
    d.qpos[grip_q] = gq
    mujoco.mj_forward(m, d)
    vf, vm = verts_in_gripper(fixed), verts_in_gripper(moving)
    print(f"\n=== gripper q={gq:+.2f} rad ===")
    for label, v in (("fixed", vf), ("moving", vm)):
        print(f"{label:6s} bbox min={np.round(v.min(0), 4)} max={np.round(v.max(0), 4)}")
    # slice along approach axis: per z-band, x/y extent of each jaw -> shows the gap between fingers
    for z_hi in (-0.04, -0.06, -0.08, -0.10):
        band = lambda v: v[(v[:, 2] <= z_hi) & (v[:, 2] > z_hi - 0.02)]
        bf, bm = band(vf), band(vm)
        fs = f"x[{bf[:, 0].min():+.3f},{bf[:, 0].max():+.3f}] y[{bf[:, 1].min():+.3f},{bf[:, 1].max():+.3f}]" if len(bf) else "-"
        ms = f"x[{bm[:, 0].min():+.3f},{bm[:, 0].max():+.3f}] y[{bm[:, 1].min():+.3f},{bm[:, 1].max():+.3f}]" if len(bm) else "-"
        print(f"  z in ({z_hi - 0.02:+.2f},{z_hi:+.2f}]  fixed {fs}   moving {ms}")
