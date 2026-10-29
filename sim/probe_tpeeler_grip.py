"""Search how the T-shaped peeler should sit in the hand so its BLADE can be dragged down the apple (kinematics).

In-hand pose: handle along the gripper y axis (head towards side*y), pinched at PINCH; the blade line is
b(gamma) = cos(gamma) x_g + sin(gamma) z_g (perpendicular to the handle). For each gamma, stroke the robot-facing
meridian at psi from lat 70 to -65 deg with targets: blade centre on the surface, blade line along the horizontal
tangent, plus a weak target keeping the handle out of the apple. Report the max blade error and min handle.normal.

Usage: python probe_tpeeler_grip.py SIDE PSI      (SIDE = 1 or -1)
"""
import sys

import mujoco
import numpy as np

from expert_peel import PINCH
from ik import PointTarget, solve_ik

side, psi = int(sys.argv[1]), float(sys.argv[2])
m = mujoco.MjModel.from_xml_path("models/so101/peel_scene.xml")
d = mujoco.MjData(m)
arm = np.array([m.jnt_qposadr[m.joint(j).id] for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")])
gb = m.body("gripper").id
C, r = np.array([0.20, 0.15, 0.1205]), 0.04
Z = np.array([0, 0, 1.0])
blade_l = PINCH + np.array([-0.005, side * 0.085, 0.003])   # blade centre in the gripper frame (as measured in sim)
back_l = blade_l - side * 0.05 * np.array([0, 1.0, 0])       # 5 cm back along the handle
out0 = np.r_[-C[:2] / np.linalg.norm(C[:2]), 0.0]
a = np.radians(psi)
n_h = np.array([np.cos(a) * out0[0] - np.sin(a) * out0[1], np.sin(a) * out0[0] + np.cos(a) * out0[1], 0.0])
tv = np.cross(Z, n_h)

seeds = [np.array(s) for s in ([0, -0.5, 0.5, 1.0, -1.5], [0, -0.5, 0.5, 1.0, 0.0], [0, -0.5, 0.5, 1.0, 1.5],
                               [0.5, -1.0, 1.0, 0.5, -2.5], [0.5, -1.0, 1.0, 0.5, 2.5])]
for gamma in range(0, 360, 15):
    g = np.radians(gamma)
    bdir_l = np.array([np.cos(g), 0.0, np.sin(g)])
    best = None
    for q0 in seeds:
        q, errs, hn = q0.copy(), [], []
        for k, lat in enumerate(np.radians(np.linspace(70, -65, 20))):
            n = np.cos(lat) * n_h + np.sin(lat) * Z
            u = -np.sin(lat) * n_h + np.cos(lat) * Z
            P = C + (r + 0.001) * n
            h = (n + u) / np.sqrt(2)                       # preferred handle: trailing out and up
            tg = [PointTarget("point", "gripper", P, 1.0, blade_l),
                  PointTarget("point", "gripper", P + 0.013 * tv, 1.0, blade_l + 0.013 * bdir_l),
                  PointTarget("point", "gripper", P + 0.05 * h, 0.1, back_l)]
            q, _ = solve_ik(m, tg, q, iters=300 if k == 0 else 150)
            d.qpos[arm] = q
            mujoco.mj_kinematics(m, d)
            Rg = d.xmat[gb].reshape(3, 3)
            e = max(np.linalg.norm(d.xpos[gb] + Rg @ blade_l - P),
                    np.linalg.norm(d.xpos[gb] + Rg @ (blade_l + 0.013 * bdir_l) - (P + 0.013 * tv)))
            errs.append(e)
            hn.append((Rg @ (back_l - blade_l)) / 0.05 @ n)
        score = max(errs) + (0.05 if min(hn) < 0.3 else 0)
        if best is None or score < best[0]:
            best = (score, max(errs), min(hn))
    print(f"side {side:+d} psi {psi:+4.0f} gamma {gamma:3d}: blade err {best[1] * 1000:5.1f} mm  handle.normal min {best[2]:+.2f}",
          flush=True)
