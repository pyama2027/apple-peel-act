"""Can the SO101 drag a held peeler's blade down the robot-facing side of the skewered apple? (kinematics only)

The peeler is assumed pinched next to the fixed finger with its handle along the gripper y axis (nominal grasp).
For each stroke (face offset psi, handle direction h) report the max IK error and robot-apple penetration.
"""
import mujoco
import numpy as np

from ik import PointTarget, solve_ik

R = 0.04
C = np.array([0.20, 0.15, 0.12])                 # apple centre when skewered (needle tip 0.1 + R - ~2 cm)
BLADE = np.array([-0.002, 0.085, -0.125])        # blade in gripper frame (nominal pinch + 8.5 cm along y)
BACK = np.array([-0.002, 0.035, -0.125])         # point 5 cm back along the handle

m = mujoco.MjModel.from_xml_path("models/so101/peel_scene.xml")
d = mujoco.MjData(m)
qa = m.jnt_qposadr[m.joint("apple_free").id]
arm = np.array([m.jnt_qposadr[m.joint(j).id] for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")])
apple_g = m.geom("apple").id

out_dir = -C[:2] / np.linalg.norm(C[:2])          # horizontal, from apple towards the robot base
rows = []
for psi in (-30, 0, 30):
    c, s = np.cos(np.radians(psi)), np.sin(np.radians(psi))
    n = np.array([c * out_dir[0] - s * out_dir[1], s * out_dir[0] + c * out_dir[1], 0.0])
    t = np.cross([0, 0, 1.0], n)
    for side in (1, -1):
        for beta in (0, 30, 60):
            h = np.cos(np.radians(beta)) * side * t + np.sin(np.radians(beta)) * np.array([0, 0, 1.0])
            q = np.array([0.0, -0.5, 0.5, 1.0, 0.0])
            errs, pens = [], []
            for phi in np.radians(np.linspace(70, -65, 28)):
                p = C + (R + 0.001) * (np.cos(phi) * n + np.sin(phi) * np.array([0, 0, 1.0]))
                q, e = solve_ik(m, [PointTarget("point", "gripper", p, 1.0, BLADE),
                                    PointTarget("point", "gripper", p + 0.05 * h, 0.5, BACK)], q, iters=150)
                d.qpos[:] = 0
                d.qpos[qa:qa + 3], d.qpos[qa + 3] = C, 1
                d.qpos[arm] = q
                mujoco.mj_forward(m, d)
                bl = d.xpos[m.body("gripper").id] + d.xmat[m.body("gripper").id].reshape(3, 3) @ BLADE
                errs.append(np.linalg.norm(bl - p))
                pen = max([-x.dist for x in d.contact[:d.ncon] if apple_g in (x.geom1, x.geom2)] + [0])
                pens.append(pen)
            rows.append((psi, side, beta, max(errs) * 1000, max(pens) * 1000))
            print(f"psi={psi:+3d} side={side:+d} beta={beta:2d}  max blade err {max(errs) * 1000:6.1f} mm"
                  f"  max robot-apple pen {max(pens) * 1000:5.1f} mm", flush=True)
