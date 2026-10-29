"""Probe SO101 kinematics: base pose, gripper site at home, and reachable workspace by random sampling."""
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "third_party/SO-ARM100/Simulation/SO101/so101_new_calib_camera.xml"

m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)
site = m.site("gripperframe").id
lo, hi = m.jnt_range[:, 0], m.jnt_range[:, 1]

for name in ("base", "shoulder", "gripper", "wrist_camera"):
    b = m.body(name).id
    mujoco.mj_forward(m, d)
    print(f"{name:12s} pos={np.round(d.xpos[b], 4)}")

for label, q in (("zeros", np.zeros(m.nq)), ("mid", (lo + hi) / 2)):
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    print(f"gripperframe @ {label}: {np.round(d.site_xpos[site], 4)}")

# pan sweep at zero pose shows which world direction "forward" is
for pan in (-1.0, 0.0, 1.0):
    d.qpos[:] = 0
    d.qpos[0] = pan
    mujoco.mj_forward(m, d)
    print(f"pan={pan:+.1f} -> gripperframe {np.round(d.site_xpos[site], 4)}")

rng = np.random.default_rng(0)
pts = []
for _ in range(20000):
    d.qpos[:] = rng.uniform(lo, hi)
    mujoco.mj_forward(m, d)
    pts.append(d.site_xpos[site].copy())
pts = np.array(pts)
low = pts[(pts[:, 2] > 0.03) & (pts[:, 2] < 0.12)]
r = np.linalg.norm(low[:, :2], axis=1)
print(f"site z in [3,12]cm: n={len(low)}  horizontal radius min={r.min():.3f} max={r.max():.3f} p90={np.percentile(r, 90):.3f}")
