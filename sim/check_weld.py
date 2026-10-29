"""Verify the weld relpose convention used by AppleTask.activate_weld: the apple must not jump when the weld turns on.
Also tries the reversed convention to show the difference."""
import mujoco
import numpy as np

from task import AppleTask

t = AppleTask()
m, d = t.m, t.d


def trial(reverse):
    t.reset(apple_xy=[0.20, 0.153])  # 3 mm off the needle axis
    d.qpos[t.apple_q + 2] = 0.11          # in the air, needle 3 cm inside the (non-colliding) apple
    mujoco.mj_forward(m, d)
    p0 = t.apple_pos()
    t.activate_weld()
    if reverse:  # body1 pose in body2 frame
        b1, b2 = t.needle_b, t.apple_b
        R2 = d.xmat[b2].reshape(3, 3)
        m.eq_data[t.weld, 3:6] = R2.T @ (d.xpos[b1] - d.xpos[b2])
        q2inv, rq = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(q2inv, d.xquat[b2])
        mujoco.mju_mulQuat(rq, q2inv, d.xquat[b1])
        m.eq_data[t.weld, 6:10] = rq
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    drift = np.linalg.norm(t.apple_pos() - p0)
    print(f"{'reversed' if reverse else 'ours    '} convention: apple moved {drift * 1000:.2f} mm in 1 s "
          f"(depth={t.insertion_depth() * 100:.1f}cm)")
    return drift


ok = trial(False)
trial(True)
print("PASS" if ok < 0.001 else "FAIL")
