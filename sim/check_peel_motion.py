"""Run the full peel expert without rendering and check the motion quality.

Reports success / peeled fraction, and per segment: max penetration of the peeler parts and the gripper into the
apple (the peeler does not collide with the apple, so penetration would only show up visually), the largest
commanded joint step per frame, and how much of the blade edge is on the skin during strokes.

Usage: python check_peel_motion.py SEED [SEED ...]
"""
import sys
from collections import defaultdict

import mujoco
import numpy as np

import expert as E
from expert_peel import PeelExpert
from run_demo import sample_scene
from task import PEEL_TOL, PeelTask


def check(seed):
    t = PeelTask()
    ex = PeelExpert(t)
    m, d = t.m, t.d
    apple_g = m.geom("apple").id
    parts = {n: m.geom(n).id for n in ("peeler_handle", "peeler_prong_a", "peeler_prong_b", "peeler_blade")}
    hand = [g for g in range(m.ngeom) if m.geom_contype[g] and
            m.geom_bodyid[g] in (m.body("gripper").id, m.body("moving_jaw_so101_v1").id)]
    pen = defaultdict(float)     # (segment, part) -> max penetration [m]
    step = defaultdict(float)    # segment -> max joint step [rad]
    cut = []
    prev = {"q": None}

    def on_frame(seg, q):
        if prev["q"] is not None:
            step[seg] = max(step[seg], np.abs(q[:5] - prev["q"][:5]).max())
        prev["q"] = q.copy()
        if not t.skewered:
            return
        for name, g in parts.items():
            pen[(seg, name)] = max(pen[(seg, name)], -mujoco.mj_geomDistance(m, d, g, apple_g, 0.05, None))
        if seg not in ("descend", "close", "insert", "release", "retreat") and not seg.startswith("twist"):
            for g in hand:
                pen[(seg, "gripper")] = max(pen[(seg, "gripper")], -mujoco.mj_geomDistance(m, d, g, apple_g, 0.05, None))
        if seg == "stroke":
            dist = np.linalg.norm(t.blade_edge() - t.apple_pos(), axis=1)
            cut.append(int((np.abs(dist - t.apple_r) < PEEL_TOL).sum()))

    t.reset(**sample_scene(np.random.default_rng(seed)), q_arm=ex.home_q())
    exc = None
    try:
        ex.run(on_frame)
    except E.GraspFailed as e:
        exc = str(e)
    cut = np.array(cut) if cut else np.zeros(1)
    worst = sorted(((v, k) for k, v in pen.items() if v > 0.003), reverse=True)[:6]
    print(f"seed {seed}: success {t.success()} peeled {t.peeled_fraction():.3f} exc {exc} | blade points on skin mean "
          f"{cut.mean():.1f}/9, >=5 in {np.mean(cut >= 5) * 100:.0f}% of stroke frames | max joint step "
          f"{np.degrees(max(step.values())):.1f} deg ({max(step, key=step.get)})", flush=True)
    for v, (seg, part) in worst:
        print(f"    penetration {v * 1000:5.1f} mm  {part:15s} in {seg}")


if __name__ == "__main__":
    for s in sys.argv[1:] or ["2000"]:
        check(int(s))
