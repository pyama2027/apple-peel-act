"""Time the two 224x224 policy cameras (s/frame) with the GL backend in MUJOCO_GL. Usage: MUJOCO_GL=egl python bench_render.py"""
import os
import time

import numpy as np

from expert import Expert
from run_demo import reset_task, sample_scene
from task import PeelTask

t = PeelTask()
reset_task(t, Expert(t).home_q(), sample_scene(np.random.default_rng(0)))
t.render()
t0 = time.time()
for _ in range(30):
    t.render()
print(os.environ.get("MUJOCO_GL", "default"), f"{(time.time() - t0) / 30:.3f} s/frame", flush=True)
os._exit(0)  # skip GL teardown (EGL free raises under WSL)
