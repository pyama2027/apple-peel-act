"""Scripted expert (IK + known object poses) for the apple-on-needle task.

The gripper is kept at a fixed tilt (TILT from vertical, top towards the base): with the SO101 wrist range a vertical
gripper is only reachable at table height (sim/check_reach.py); 50 deg reaches every waypoint.
Cartesian segments are interpolated with a minimum-jerk profile and IK is re-solved every frame (warm-started).
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ik import grasp_targets, gripper_axis, solve_ik
from task import FPS, AppleTask

TILT = np.radians(50)
OPEN, CLOSED = 1.5, -0.17           # gripper joint targets [rad]
APPROACH_D = 0.09                    # pre-grasp stand-off along the gripper axis [m] (fingers slide in along their length)
Z_LIFT = 0.18                        # grasp-site height while carrying [m]
INSERT_DEPTH = 0.03                  # aim for 3 cm of needle inside the apple (weld triggers at 2 cm)
HOME_P = np.array([0.20, 0.0, 0.18])
MIN_LIFT = 0.03                      # apple must rise this much after "lift", otherwise the grasp failed
MAX_JOINT_STEP = np.radians(4)       # per control frame (120 deg/s): when the IK target set changes, the solution can
                                     # jump to another arm configuration; the command moves there smoothly instead


class GraspFailed(RuntimeError):
    pass


def min_jerk(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


@dataclass
class Segment:
    name: str
    duration: float
    p_to: object = None                 # controlled-point target at the end: array, callable() -> array (re-evaluated
                                        # every frame, for closed-loop alignment), or None = hold position
    grip_to: float | None = None        # gripper target at the end (None = hold)
    joint_to: np.ndarray | None = None  # joint-space segment (used for the return home)
    turn_to: float | None = None        # turntable target at the end [rad] (PeelTask only; None = hold)
    path: Callable | None = None        # path(s in [0,1]) -> point; overrides straight-line interpolation of p_to
    targets: Callable | None = None     # targets(p, s) -> IK PointTargets for the controlled point (default: grasp site)


@dataclass
class Expert:
    task: AppleTask
    log: dict = field(default_factory=dict)

    def ik(self, targets, q_init):
        q, err = solve_ik(self.task.m, targets, q_init, iters=100)
        self.log["max_ik_err"] = max(self.log.get("max_ik_err", 0.0), err)
        return q

    def home_q(self):
        q = self.ik(grasp_targets(HOME_P, tilt=TILT), np.zeros(5))
        return np.r_[q, 0.3]

    def start(self):
        q = self.task.qpos()
        self.q_arm, self.grip = q[:5], q[5]
        self.turn = q[6] if len(q) > 6 else None
        self.p = HOME_P.copy()

    def play(self, seg, on_frame):
        """on_frame(segment_name, action_q) is called once per control frame BEFORE the action is applied, so the sim
        state at that moment is the observation paired with action_q (same ordering as lerobot-record)."""
        t = self.task
        targets = seg.targets or (lambda p, s: grasp_targets(p, tilt=TILT))
        n = max(1, int(round(seg.duration * FPS)))
        p0, g0, q0, r0 = self.p.copy(), self.grip, self.q_arm.copy(), self.turn
        for k in range(n):
            s = min_jerk((k + 1) / n)
            if seg.joint_to is not None:
                self.q_arm = (1 - s) * q0 + s * seg.joint_to[:5]
            else:
                if seg.path is not None:
                    target = seg.path(s)
                else:
                    p_end = seg.p_to() if callable(seg.p_to) else seg.p_to
                    target = p0 if p_end is None else (1 - s) * p0 + s * p_end
                q_ik = self.ik(targets(target, s), self.q_arm)
                self.q_arm = self.q_arm + np.clip(q_ik - self.q_arm, -MAX_JOINT_STEP, MAX_JOINT_STEP)
            g = g0 if seg.grip_to is None else (1 - s) * g0 + s * seg.grip_to
            q = np.r_[self.q_arm, g]
            if r0 is not None:
                q = np.r_[q, r0 if seg.turn_to is None else (1 - s) * r0 + s * seg.turn_to]
            on_frame(seg.name, q)
            t.step(q)
            self.after_frame(seg)
        if seg.path is not None:
            self.p = np.asarray(seg.path(1.0), dtype=float).copy()
        elif seg.p_to is not None:
            self.p = np.asarray(seg.p_to() if callable(seg.p_to) else seg.p_to, dtype=float).copy()
        if seg.grip_to is not None:
            self.grip = seg.grip_to
        if seg.turn_to is not None:
            self.turn = seg.turn_to

    def after_frame(self, seg):
        t = self.task
        if seg.name == "insert":
            dxy = np.linalg.norm(t.apple_pos()[:2] - t.needle_tip_pos()[:2])
            self.log.setdefault("insert_trace", []).append(
                (round(dxy * 1000, 1), round(t.insertion_depth() * 1000, 1)))

    def skewer(self, on_frame, max_attempts=3):
        """Pick the apple from the table and push it onto the needle, release, back off along the gripper axis."""
        t = self.task
        needle = t.needle_tip_pos()

        for attempt in range(max_attempts):
            apple0 = t.apple_pos()
            at_apple = apple0.copy()
            pregrasp = at_apple + APPROACH_D * gripper_axis(at_apple, TILT)
            pfx = "" if attempt == 0 else f"r{attempt}_"
            self.play(Segment(pfx + "approach", 2.0, pregrasp, OPEN), on_frame)
            self.play(Segment(pfx + "descend", 1.5, at_apple), on_frame)
            self.play(Segment(pfx + "close", 1.0, grip_to=CLOSED), on_frame)
            self.play(Segment(pfx + "lift", 1.5, np.array([at_apple[0], at_apple[1], Z_LIFT])), on_frame)
            rise = t.apple_pos()[2] - apple0[2]
            self.log["lift_rise_mm"] = round(rise * 1000, 1)
            if rise >= MIN_LIFT:
                break
            if attempt == max_attempts - 1:
                raise GraspFailed(f"apple rose only {rise * 1000:.0f} mm after {max_attempts} attempts")
            # open and back off before retrying with the apple's new position
            self.play(Segment(pfx + "open_retry", 0.5, grip_to=OPEN), on_frame)
            self.play(Segment(pfx + "retreat_retry", 1.5, apple0 + APPROACH_D * gripper_axis(apple0, TILT)), on_frame)

        # The apple sits a little off the grasp site, and that offset rotates with the hand while carrying.
        # Re-measure it every frame (privileged sim state) so the apple centre, not the site, ends over the needle.
        site_id = t.m.site("grasp").id
        offset = lambda: t.apple_pos() - t.d.site_xpos[site_id]
        self.log["grasp_offset_mm"] = np.round(offset() * 1000, 1).tolist()
        apple_z_inserted = needle[2] + t.apple_r - INSERT_DEPTH
        over = lambda: np.array([needle[0], needle[1], Z_LIFT]) - [*offset()[:2], 0]
        down = lambda: np.array([needle[0], needle[1], apple_z_inserted]) - offset()
        self.play(Segment("transfer", 2.5, over), on_frame)
        self.play(Segment("insert", 2.0, down), on_frame)
        down = self.p.copy()
        self.log["skewered_at_insert_end"] = t.skewered
        self.play(Segment("release", 1.0, grip_to=OPEN), on_frame)
        self.play(Segment("retreat", 1.5, down + APPROACH_D * gripper_axis(down, TILT)), on_frame)

    def run(self, on_frame):
        self.start()
        self.skewer(on_frame)
        self.play(Segment("home", 2.0, joint_to=self.home_q()), on_frame)
