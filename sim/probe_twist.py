"""How far can the SO101 turn the skewered apple about the (vertical) needle axis while holding it? (kinematics only)

Start from the equator grasp at the apple centre with the gripper tilted by `tilt`, then ask IK to rotate the whole
gripper rigidly about the vertical line through the apple centre by phi (3 gripper points as targets).
"""
import mujoco
import numpy as np

from ik import PointTarget, grasp_targets, solve_ik

C = np.array([0.20, 0.15, 0.12])
m = mujoco.MjModel.from_xml_path("models/so101/peel_scene.xml")
d = mujoco.MjData(m)
arm = np.array([m.jnt_qposadr[m.joint(j).id] for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")])
LOCAL = [np.array([0.034, 0, -0.113]), np.array([0.034, 0, -0.043]), np.array([-0.02, 0, -0.113])]  # grasp, axis, fixed finger


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def world_points(q):
    d.qpos[arm] = q
    mujoco.mj_kinematics(m, d)
    b = m.body("gripper").id
    return [d.xpos[b] + d.xmat[b].reshape(3, 3) @ l for l in LOCAL]


for tilt in (40, 50, 60):
    for roll_hint in (0.0,):
        q0, e0 = solve_ik(m, grasp_targets(C, tilt=np.radians(tilt)), np.array([0, -0.5, 0.5, 1.0, roll_hint]), iters=300)
        P0 = world_points(q0)
        res = []
        for sign in (1, -1):
            q, ok = q0.copy(), 0
            for phi in range(5, 95, 5):
                a = np.radians(sign * phi)
                tg = [PointTarget("point", "gripper", C + Rz(a) @ (p - C), w, l) for p, l, w in zip(P0, LOCAL, (1, 0.7, 0.7))]
                q, e = solve_ik(m, tg, q, iters=200)
                if e > 0.003:
                    break
                ok = sign * phi
            res.append(ok)
        print(f"tilt {tilt}: base grasp err {e0 * 1000:.1f} mm, rigid twist range {res[1]:+d} .. {res[0]:+d} deg", flush=True)
