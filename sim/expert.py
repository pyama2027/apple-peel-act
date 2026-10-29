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


HELD_DIST = 0.05                     # apple centre this close to the grasp site = still in the hand [m]
MAX_DROPS = 2                        # pick the apple up again at most this many times
MAX_INSERTS = 5                      # lift and re-aim at most this many times when the apple sits on the needle tip
MISS_PRESS = 0.004                   # a planned miss presses the apple this far onto the tip [m]
SETTLE_SPEED, SETTLE_MAX = 0.005, 5.0  # after a drop, wait until the apple moves slower than this [m/s], at most [s]


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
    drop_frac: float | None = None      # drop the apple on purpose this far into the transfer (None = never)
    miss_plan: list = field(default_factory=list)   # xy aim offsets [m] for the first inserts (planned misses)

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

    def grasp_apple(self, on_frame, pfx="", max_attempts=3):
        """Pick the apple up from wherever it lies on the table (retrying with its new position if the lift fails)."""
        t = self.task
        for attempt in range(max_attempts):
            apple0 = t.apple_pos()
            self.log.setdefault("grasp_at", []).append(np.round(apple0[:2], 3).tolist())
            at_apple = apple0.copy()
            pregrasp = at_apple + APPROACH_D * gripper_axis(at_apple, TILT)
            p = pfx + ("" if attempt == 0 else f"r{attempt}_")
            self.play(Segment(p + "approach", 2.0, pregrasp, OPEN), on_frame)
            self.play(Segment(p + "descend", 1.5, at_apple), on_frame)
            self.play(Segment(p + "close", 1.0, grip_to=CLOSED), on_frame)
            self.play(Segment(p + "lift", 1.5, np.array([at_apple[0], at_apple[1], Z_LIFT])), on_frame)
            rise = t.apple_pos()[2] - apple0[2]
            self.log["lift_rise_mm"] = round(rise * 1000, 1)
            if rise >= MIN_LIFT:
                return
            if attempt == max_attempts - 1:
                raise GraspFailed(f"apple rose only {rise * 1000:.0f} mm after {max_attempts} attempts")
            # open and back off before retrying with the apple's new position
            self.play(Segment(p + "open_retry", 0.5, grip_to=OPEN), on_frame)
            self.play(Segment(p + "retreat_retry", 1.5, apple0 + APPROACH_D * gripper_axis(apple0, TILT)), on_frame)

    def apple_held(self):
        t = self.task
        return np.linalg.norm(t.apple_pos() - t.d.site_xpos[t.m.site("grasp").id]) < HELD_DIST

    def apple_speed(self):
        t = self.task
        adr = t.m.jnt_dofadr[t.m.joint("apple_free").id]
        return float(np.linalg.norm(t.d.qvel[adr:adr + 3]))

    def recover_drop(self, on_frame, pfx):
        """The apple fell out of the hand: open, go back home (approaching the apple from right above it pushes it
        away), wait until it lies still, then pick it up again from where it ended up."""
        self.log.setdefault("drops", []).append(np.round(self.task.apple_pos(), 3).tolist())
        self.play(Segment(pfx + "drop_open", 0.4, grip_to=OPEN), on_frame)
        self.play(Segment(pfx + "drop_home", 1.5, joint_to=self.home_q()), on_frame)
        self.p = HOME_P.copy()
        for _ in range(int(SETTLE_MAX / 0.2)):
            if self.apple_speed() < SETTLE_SPEED:
                break
            self.play(Segment(pfx + "drop_wait", 0.2), on_frame)
        self.grasp_apple(on_frame, pfx)

    def insert_until_skewered(self, on_frame, needle, offset, over, pfx):
        """Push the apple down onto the needle. If it ends up resting on the needle tip (off-centre: a miss, on purpose
        when self.miss_plan says so), lift it back up, re-aim with the re-measured offset and try again.
        Returns True when skewered, False when the apple was dropped."""
        t = self.task
        z_in = needle[2] + t.apple_r - INSERT_DEPTH
        for k in range(MAX_INSERTS):
            miss = self.miss_plan.pop(0) if self.miss_plan else None
            if miss is None:
                down = lambda: np.array([needle[0], needle[1], z_in]) - offset()
            else:  # aim off-centre; the apple stops on the tip, pressed MISS_PRESS onto it
                down = lambda m=np.asarray(miss): np.r_[needle[:2] + m, needle[2] + t.apple_r - MISS_PRESS] - offset()
            self.play(Segment(pfx + ("insert" if k == 0 else f"reinsert{k}"), 2.0, down), on_frame)
            if t.skewered:
                return True
            if not self.apple_held():
                return False
            self.log.setdefault("missed_skewer", []).append(
                round(float(np.linalg.norm(t.apple_pos()[:2] - t.needle_tip_pos()[:2])) * 1000, 1))
            self.play(Segment(pfx + f"lift_off{k + 1}", 1.2, over), on_frame)
        raise GraspFailed(f"not skewered after {MAX_INSERTS} inserts")

    def skewer(self, on_frame, max_attempts=3):
        """Pick the apple from the table and push it onto the needle, release, back off along the gripper axis.
        If the apple is dropped on the way (on purpose when self.drop_frac is set: the gripper opens that far into the
        transfer, so the dataset shows recoveries), it is picked up again from where it landed."""
        t = self.task
        needle = t.needle_tip_pos()
        self.grasp_apple(on_frame, max_attempts=max_attempts)

        site_id = t.m.site("grasp").id
        apple_z_inserted = needle[2] + t.apple_r - INSERT_DEPTH
        for trial in range(MAX_DROPS + 1):
            pfx = "" if trial == 0 else f"d{trial}_"
            # The apple sits a little off the grasp site, and that offset rotates with the hand while carrying.
            # Re-measure it every frame (privileged sim state) so the apple centre, not the site, ends over the needle.
            offset = lambda: t.apple_pos() - t.d.site_xpos[site_id]
            self.log["grasp_offset_mm"] = np.round(offset() * 1000, 1).tolist()
            over = lambda: np.array([needle[0], needle[1], Z_LIFT]) - [*offset()[:2], 0]
            if trial == 0 and self.drop_frac is not None:
                p0, f = self.p.copy(), self.drop_frac
                self.play(Segment("transfer", 2.5 * f, lambda: p0 + f * (over() - p0)), on_frame)
                self.play(Segment("let_go", 0.3, grip_to=OPEN), on_frame)
                self.recover_drop(on_frame, f"d{trial + 1}_")
                continue
            self.play(Segment(pfx + "transfer", 2.5, over), on_frame)
            if not self.apple_held():
                if trial == MAX_DROPS:
                    raise GraspFailed("apple dropped too often")
                self.recover_drop(on_frame, f"d{trial + 1}_")
                continue
            if self.insert_until_skewered(on_frame, needle, offset, over, pfx) or trial == MAX_DROPS:
                break
            self.recover_drop(on_frame, f"d{trial + 1}_")
        down = self.p.copy()
        self.log["skewered_at_insert_end"] = t.skewered
        self.play(Segment("release", 1.0, grip_to=OPEN), on_frame)
        self.play(Segment("retreat", 1.5, down + APPROACH_D * gripper_axis(down, TILT)), on_frame)

    def run(self, on_frame):
        self.start()
        self.skewer(on_frame)
        self.play(Segment("home", 2.0, joint_to=self.home_q()), on_frame)
