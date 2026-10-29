"""For each waypoint height, the smallest gripper tilt (from vertical, top towards the base) that IK can reach
over the whole apple / needle region."""
import numpy as np

from ik import grasp_targets, solve_ik
from task import AppleTask

t = AppleTask()
m = t.m
needle = t.needle_tip_pos()  # (0.20, 0.15, 0.10)
r = t.apple_r
TILTS = np.radians(np.arange(0, 75, 5))

regions = {
    "apple": [(x, y) for x in (0.15, 0.20, 0.25) for y in (-0.20, -0.15, -0.10)],
    "needle": [(needle[0] + dx, needle[1] + dy) for dx in (-0.02, 0, 0.02) for dy in (-0.02, 0, 0.02)],
}
heights = {
    "apple": {"grasp": 0.040, "pregrasp": 0.130, "lift": 0.180},
    "needle": {"above": needle[2] + r + 0.015, "inserted": needle[2] + r - 0.030},
}
for region, pts in regions.items():
    for label, z in heights[region].items():
        line = []
        for tilt in TILTS:
            errs = [solve_ik(m, grasp_targets([x, y, z], tilt=tilt), np.zeros(5), iters=300)[1] for x, y in pts]
            n_ok = int(np.sum(np.array(errs) < 0.002))
            line.append(f"{np.degrees(tilt):.0f}:{n_ok}")
        print(f"{region:6s} {label:9s} z={z:.3f}  tilt_deg:reachable/9 -> " + " ".join(line))
