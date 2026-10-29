"""Physics test of turning the skewered apple by regrasping (PeelExpert.twist_cycle): turntable angle per cycle."""
import sys

import numpy as np

from expert import HOME_P
from expert_peel import PeelExpert, TWIST_FROM
from run_demo import sample_scene
from task import PeelTask

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
task = PeelTask()
ex = PeelExpert(task)
task.reset(**sample_scene(np.random.default_rng(seed)), q_arm=ex.home_q())
ex.start()
ex.skewer(lambda seg, q: None)
print("skewered", task.skewered, "turntable %.1f deg" % np.degrees(task.turntable_angle()))
a = TWIST_FROM
for k in range(n):
    a0 = task.turntable_angle()
    a = ex.twist_cycle(lambda seg, q: None, a)
    print(f"cycle {k}: turned {np.degrees(task.turntable_angle() - a0):+6.1f} deg  total {np.degrees(task.turntable_angle()):+6.1f}"
          f"  max ik err {ex.log['max_ik_err'] * 1000:.1f} mm  apple-needle dxy {np.linalg.norm(task.apple_pos()[:2] - task.needle_tip_pos()[:2]) * 1000:.1f} mm", flush=True)
