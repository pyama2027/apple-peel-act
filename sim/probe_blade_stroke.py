"""Can the held (T-shaped) peeler run its BLADE along the robot-facing side of the skewered apple? (kinematics)

The peeler pose in the gripper is taken from the sim right after the expert lifts it off the pegs. For a stroke down
the meridian at psi, the targets are: blade centre on the surface, blade line along the horizontal tangent t (blade
perpendicular to the stroke), and the handle trailing behind the blade: from the blade back along
h = cos(alpha) n + sin(alpha) u (n = outward normal, u = upward meridian tangent). Report the max IK error.
"""
import sys

import numpy as np

import expert as E
from expert_peel import PeelExpert
from ik import PointTarget, solve_ik
from run_demo import sample_scene
from task import PeelTask


class Stop(Exception):
    pass


orig = E.Expert.play


def play(self, seg, on_frame):
    orig(self, seg, on_frame)
    if seg.name == "lift_peeler":
        raise Stop


E.Expert.play = play
t = PeelTask()
ex = PeelExpert(t)
t.reset(**sample_scene(np.random.default_rng(2000)), q_arm=ex.home_q())
try:
    ex.run(lambda s, q: None)
except Stop:
    pass
m, d = t.m, t.d
x, R = ex.gripper_frame()
Rp = d.xmat[t.peeler_b].reshape(3, 3)
blade_l = R.T @ (t.blade_pos() - x)          # blade centre in the gripper frame
bdir_l = R.T @ Rp[:, 1]                       # blade line direction (peeler y)
hdir_l = -(R.T @ Rp[:, 0])                    # from blade back along the handle (peeler -x)
print("blade in gripper mm", np.round(blade_l * 1000, 1), "blade dir", np.round(bdir_l, 2), "handle back dir", np.round(hdir_l, 2))

C = t.apple_pos()
r = t.apple_r
q_start = ex.q_arm.copy()
out0 = np.r_[-C[:2] / np.linalg.norm(C[:2]), 0.0]
psis = [float(a) for a in sys.argv[1:]] or [-30, -15, 0, 15, 30]
for psi in psis:
    a = np.radians(psi)
    n_h = np.array([np.cos(a) * out0[0] - np.sin(a) * out0[1], np.sin(a) * out0[0] + np.cos(a) * out0[1], 0.0])
    tvec = np.cross([0, 0, 1.0], n_h)
    for sgn in (1, -1):
        for alpha in (20, 35, 50, 65, 80):
            q = q_start.copy()
            errs = []
            for lat in np.radians(np.linspace(70, -65, 24)):
                n = np.cos(lat) * n_h + np.sin(lat) * np.array([0, 0, 1.0])
                u = -np.sin(lat) * n_h + np.cos(lat) * np.array([0, 0, 1.0])
                P = C + (r + 0.001) * n
                h = np.cos(np.radians(alpha)) * n + np.sin(np.radians(alpha)) * u
                tg = [PointTarget("point", "gripper", P, 1.0, blade_l),
                      PointTarget("point", "gripper", P + sgn * 0.013 * tvec, 1.0, blade_l + 0.013 * bdir_l),
                      PointTarget("point", "gripper", P + 0.05 * h, 0.5, blade_l + 0.05 * hdir_l)]
                q, e = solve_ik(m, tg, q, iters=200)
                errs.append(e)
            print(f"psi {psi:+4.0f} blade sign {sgn:+d} alpha {alpha:2d}: max err {max(errs) * 1000:6.1f} mm"
                  f"  (top {errs[0] * 1000:5.1f}, mid {errs[12] * 1000:5.1f}, bottom {errs[-1] * 1000:5.1f})", flush=True)
