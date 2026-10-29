"""Scripted expert for the full task: skewer the apple, then for each of 4 faces: pick up the peeler, peel one side in
vertical blade strokes, put the peeler back, and turn the apple 90 deg by hand (several regrasps).

Strokes cut with the BLADE of the T-shaped peeler: the blade edge stays level (along the horizontal tangent) and on the
skin while it is dragged down a meridian, with the handle trailing out and up. Only some in-hand peeler poses allow
this with a 5-DoF arm (sim/probe_tpeeler_grip.py): with the head on the gripper's -y side (peeler stored head towards
the robot's right) the strokes at psi = -100..-55 deg from the robot-facing direction are reachable (< 3 mm blade
error, handle outside the apple); strokes nearer the robot-facing side are not.

Turning the apple by hand (sim/probe_twist.py, sim/probe_twist_sim.py): holding the apple at its equator with the jaw
line horizontal, the 5-DoF arm can only turn the jaw line about the needle axis over a small range; rotation about the
jaw line itself is left free (the fingers spin on their contacts). Jaw +x along -tangent, gripper tilted 45 deg:
0..+30 deg, with the wrist roll near -1.7 rad, reachable from the skewer pose (roll ~ -2.7; the +tangent start needs
roll +1.65, on the far side of the joint range). Each grasp turns the apple ~26 deg; the last one only the remainder.

Whenever the IK target set changes (grasp site / peeler pinch / blade / twist), the orientation targets are blended
from the current gripper pose over the first segment, so the commanded joints never jump.
"""
import os

import numpy as np

from expert import APPROACH_D, CLOSED, OPEN, SETTLE_MAX, SETTLE_SPEED, TILT, Expert, GraspFailed, Segment  # noqa: F401
from ik import PointTarget, gripper_axis

# peeler grasp
PINCH = np.array([-0.002, 0.0, -0.125])   # gripper frame: thin handle pinched against the fixed finger extension
HEAD_SIDE = -1                             # peeler head on the gripper's -y side: then the T blade can stay level and on
                                           # the surface in a vertical stroke (sim/probe_tpeeler_grip.py); with +y it cannot
PEELER_LIFT = 0.08
CLOSED_PEELER = -0.05                       # finger gap ~14 mm on the 12-14 mm handle: ~2.5 mm contact penetration
                                           # (fully closed = -0.17 sank the handle 8 mm into the fingers)
MIN_PEELER_RISE = 0.04
CLEAR_RISE = 0.06                          # rise before heading to the peeler
PLACE_CLEAR = -0.002                       # press the peeler 2 mm onto the pegs before releasing (dropped from above
                                           # while rolled in the grip, it tumbles off)

# peeling strokes
N_FACES = 4
STROKE_PSI = (-95.0, -75.0, -55.0)         # stroke meridians relative to the robot-facing direction [deg]; together
                                           # with the 36 mm blade they cover a 90 deg face
BLADE_SIGN = -1                            # peeler +y (along the blade) points along -tangent
HANDLE_UP = np.radians(30)                 # preferred handle direction: from the outward normal, 30 deg up
LAT_TOP, LAT_BOTTOM = 72.0, -66.0          # stroke latitude range [deg]
SKIN_CLEAR = 0.001                         # blade edge runs this far outside the flesh radius
STANDOFF = 0.025                           # blade distance from the surface between strokes
BLADE_REF = 0.015                          # second IK point this far along the blade edge
BACK_L = 0.05                              # third (weak) IK point this far back along the handle
RAISE = 0.08                               # lift the peeler this far after the last stroke of a face

# turning the apple
GRASP_LOCAL = np.array([0.034, 0, -0.113])        # "grasp" site in the gripper frame (apple centre when held)
JAW_LOCAL = GRASP_LOCAL + [0.04, 0, 0]            # 4 cm along the jaw closing direction
AXIS_LOCAL = GRASP_LOCAL + [0, 0, 0.07]           # 7 cm up the gripper axis
TWIST_TILT = np.radians(45)
TWIST_FROM, TWIST_MAX = 0.0, 27.0                 # jaw-line angle from the -tangential direction [deg]
TURN_TOL = 3.0                                    # stop regrasping when within this of the face angle [deg]
TURN_BIAS = 3.0                                   # grasping/releasing turns the apple ~1-2 deg extra: command less
MAX_TWISTS = 7
TABLE_PINCH = PINCH + [0, 0, -0.016]       # the fingertips reach 20 mm past PINCH: for a handle lying on the table
                                           # (12 mm tall) pinch near the tips instead
TABLE_PINCH_Z = float(os.environ.get("TABLE_PINCH_Z", "0.0"))     # pinch this far above the handle centre
PEELER_SPOT = np.array([0.26, -0.03])        # where a dropped peeler is laid down to be re-taken with the usual grasp
SPOT_TILT = np.radians(float(os.environ.get("SPOT_TILT", "15")))  # re-grasp tilt at the spot: at the usual 50 deg the
                                             # fingertips hit the table; the peeler then sits rolled by 50-15 deg in the
                                             # hand and settles flat again when put on the pegs
SPOT_CLEAR = 0.002                           # lower it to 2 mm above resting before letting go
TABLE_ABOVE = 0.08                                              # top-down approach from this far above the handle
MAX_PEELER_DROPS = 2                              # pick the peeler up again at most this many times per face


def unit(v):
    return v / np.linalg.norm(v)


def blend(a, b):
    """s -> unit vector going from a to b."""
    return lambda s: unit((1 - s) * a + s * b)


class PeelExpert(Expert):
    peeler_drop: tuple | None = None     # (face, fraction): drop the peeler on purpose on the way to that face
    # ---------- geometry helpers ----------
    def gripper_frame(self):
        d, b = self.task.d, self.task.m.body("gripper").id
        return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()

    def point_world(self, local):
        x, R = self.gripper_frame()
        return x + R @ local

    def to_local(self, p):
        x, R = self.gripper_frame()
        return R.T @ (p - x)

    def site(self, name):
        return self.task.d.site_xpos[self.task.m.site(name).id].copy()

    def rot_about_apple(self, v, deg):
        a = np.radians(deg)
        return np.array([np.cos(a) * v[0] - np.sin(a) * v[1], np.sin(a) * v[0] + np.cos(a) * v[1], v[2]])

    def facing(self, psi_deg):
        """Horizontal unit vector from the apple towards the robot base, rotated by psi."""
        c = self.task.apple_pos()
        return self.rot_about_apple(np.r_[-c[:2] / np.linalg.norm(c[:2]), 0.0], psi_deg)

    def blade_dir(self, psi_deg):
        """Wanted direction of the peeler's +y (blade) axis for a stroke at psi: level, tangent to the apple."""
        return BLADE_SIGN * np.cross([0, 0, 1.0], self.facing(psi_deg))

    def handle_pref(self, p):
        """Preferred handle direction (blade -> grip) with the blade at p: trailing out from the skin and up."""
        n = unit(p - self.task.apple_pos())
        u = unit(np.array([0, 0, 1.0]) - n[2] * n)
        return np.cos(HANDLE_UP) * n + np.sin(HANDLE_UP) * u

    def surface(self, lat_deg, rho, psi_deg):
        lat = np.radians(lat_deg)
        return self.task.apple_pos() + rho * (np.cos(lat) * self.facing(psi_deg) + np.sin(lat) * np.array([0, 0, 1.0]))

    def jaw_dir(self, a_deg):
        """Horizontal jaw-line direction for turning the apple: -tangential at the apple, rotated by a about z."""
        c = self.task.apple_pos()
        return self.rot_about_apple(np.cross(np.r_[c[:2] / np.linalg.norm(c[:2]), 0.0], [0, 0, 1.0]), a_deg)

    # ---------- IK target sets ----------
    def pinch_targets(self, axis_of_s, hdir_of_s, pinch=PINCH):
        """Pinch point at p, gripper axis along axis(s), peeler grip -> blade direction hdir(s) along HEAD_SIDE * y."""
        return lambda p, s: [PointTarget("point", "gripper", p, 1.0, pinch),
                             PointTarget("point", "gripper", p + 0.07 * axis_of_s(s), 0.5, pinch + [0, 0, 0.07]),
                             PointTarget("point", "gripper", p + 0.03 * hdir_of_s(s), 0.3, pinch + [0, 0.03 * HEAD_SIDE, 0])]

    def peeler_y(self):
        return self.task.d.xmat[self.task.peeler_b].reshape(3, 3)[:, 1].copy()

    def blade_targets(self, bdir_of_s, h_of_s=None, w_handle=0.1):
        """Blade centre at p, blade edge along bdir(s), and (weakly) the handle along h(s, p). The peeler's pose in the
        gripper frame is re-measured every frame from the sim (it can shift slightly in the grip)."""
        h_of_s = h_of_s or (lambda s, p: self.handle_pref(p))

        def f(p, s):
            blade, grip = self.task.blade_pos(), self.site("peeler_grip")
            edge = blade + BLADE_REF * self.peeler_y()
            back = blade + BACK_L * unit(grip - blade)
            return [PointTarget("point", "gripper", p, 1.0, self.to_local(blade)),
                    PointTarget("point", "gripper", p + BLADE_REF * bdir_of_s(s), 1.0, self.to_local(edge)),
                    PointTarget("point", "gripper", p + BACK_L * h_of_s(s, p), w_handle, self.to_local(back))]
        return f

    def twist_targets(self, a_of_s, axis_of_s=None, w_axis=0.5):
        """grasp site at p, jaw line along jaw_dir(a(s)); optionally the gripper axis along axis(s)."""
        def f(p, s):
            tg = [PointTarget("point", "gripper", p, 1.0, GRASP_LOCAL),
                  PointTarget("point", "gripper", p + 0.04 * self.jaw_dir(a_of_s(s)), 1.0, JAW_LOCAL)]
            if axis_of_s is not None:
                tg.append(PointTarget("point", "gripper", p + 0.07 * axis_of_s(s), w_axis, AXIS_LOCAL))
            return tg
        return f

    def current_jaw_angle(self):
        """Angle of the current (horizontal projection of the) jaw line relative to jaw_dir(0) [deg]."""
        x = self.gripper_frame()[1][:, 0]
        j0 = self.jaw_dir(0.0)
        return np.degrees(np.arctan2(np.cross(j0, x)[2], j0 @ x))

    def current_handle_dir(self):
        return unit(self.site("peeler_grip") - self.task.blade_pos())

    def blend_from_current(self, psi, start):
        """Blade / handle direction targets going from the current peeler pose to the stroke pose at `start`."""
        bdir = blend(self.peeler_y(), self.blade_dir(psi))
        h0, h1 = self.current_handle_dir(), self.handle_pref(start)
        return bdir, (lambda s, p: unit((1 - s) * h0 + s * h1))

    # ---------- peeler ----------
    def pick_peeler(self, on_frame, pinch=PINCH, name="", tilt=TILT):
        t = self.task
        grip0 = self.site("peeler_grip")
        hdir = unit(t.blade_pos() - grip0)
        if not hasattr(self, "peeler_home"):  # the rest pose on the pegs; every put-back aims here (no drift build-up)
            self.peeler_home = (grip0, hdir)
        _, R = self.gripper_frame()
        ax = gripper_axis(grip0, tilt)
        self.p = self.point_world(pinch)
        # straight up first: going directly from beside the apple to the peeler brushes the apple with the open jaws
        hold = self.pinch_targets(lambda s: R[:, 2], lambda s: HEAD_SIDE * R[:, 1], pinch)
        self.play(Segment(name + "rise", 0.8, self.p + [0, 0, CLEAR_RISE], targets=hold), on_frame)
        self.play(Segment(name + "to_peeler", 2.0, grip0 + APPROACH_D * ax, OPEN,
                          targets=self.pinch_targets(blend(R[:, 2], ax), blend(HEAD_SIDE * R[:, 1], hdir), pinch)), on_frame)
        tg = self.pinch_targets(lambda s: ax, lambda s: hdir, pinch)
        self.play(Segment(name + "descend_peeler", 1.5, grip0, targets=tg), on_frame)
        if name:
            self.log.setdefault(name + "pick_err_mm", []).append(np.round((self.point_world(pinch) - grip0) * 1000, 1).tolist())
        self.play(Segment(name + "close_peeler", 1.0, grip_to=CLOSED_PEELER, targets=tg), on_frame)
        z0 = t.d.xpos[t.peeler_b][2]
        self.play(Segment(name + "lift_peeler", 1.2, grip0 + [0, 0, PEELER_LIFT], targets=tg), on_frame)
        rise = t.d.xpos[t.peeler_b][2] - z0
        self.log.setdefault("peeler_rise_mm", []).append(round(rise * 1000, 1))
        if rise < MIN_PEELER_RISE:
            raise GraspFailed(f"peeler rose only {rise * 1000:.0f} mm")

    def put_peeler(self, on_frame):
        grip0, hdir = self.peeler_home
        _, R = self.gripper_frame()
        ax = gripper_axis(grip0, TILT)
        self.p = self.point_world(PINCH)
        # the peeler slips a little in the grip: aim its grip site (not the pinch point) at the rest pose, with the
        # offset re-measured every frame
        offset = lambda: self.site("peeler_grip") - self.point_world(PINCH)
        self.play(Segment("to_stand", 2.0, lambda: grip0 + [0, 0, PEELER_LIFT] - offset(),
                          targets=self.pinch_targets(blend(R[:, 2], ax), blend(HEAD_SIDE * R[:, 1], hdir))), on_frame)
        tg = self.pinch_targets(lambda s: ax, lambda s: hdir)
        self.play(Segment("place_peeler", 1.2, lambda: grip0 + [0, 0, PLACE_CLEAR] - offset(), targets=tg), on_frame)
        self.play(Segment("release_peeler", 0.6, grip_to=OPEN, targets=tg), on_frame)
        self.play(Segment("leave_peeler", 0.8, self.p + APPROACH_D * ax, targets=tg), on_frame)

    def peeler_held(self):
        return np.linalg.norm(self.site("peeler_grip") - self.point_world(PINCH)) < 0.04

    def peeler_speed(self):
        t = self.task
        adr = t.m.jnt_dofadr[t.m.joint("peeler_free").id]
        return float(np.linalg.norm(t.d.qvel[adr:adr + 3]))

    def drop_peeler(self, on_frame, frac):
        """Carry the peeler `frac` of the way towards the apple, then let go of it (dataset diversity: the policy
        should see the peeler fall and be picked up again from the table)."""
        t = self.task
        psi0 = STROKE_PSI[0]
        self.p = t.blade_pos()
        start = self.surface(LAT_TOP + 5, t.apple_r + STANDOFF, psi0)
        bdir, hdir = self.blend_from_current(psi0, start)
        self.play(Segment("to_apple", 2.5 * frac, self.p + frac * (start - self.p),
                          targets=self.blade_targets(lambda s: bdir(frac * s), lambda s, p: hdir(frac * s, p))), on_frame)
        self.play(Segment("let_go_peeler", 0.3, grip_to=OPEN, joint_to=np.r_[self.q_arm]), on_frame)

    def pick_peeler_from_table(self, on_frame):
        """Pick the peeler up from wherever it lies on the table: gripper pointing straight down (reachable at table
        height), jaws across the handle, the head on the same side of the hand as in the normal grasp, so the
        in-hand pose is the usual one and put_peeler() can set it back on the pegs."""
        t = self.task
        grip0 = self.site("peeler_grip")
        h = t.blade_pos() - grip0
        hdir = unit(np.r_[h[:2], 0.0])
        down = np.array([0, 0, 1.0])
        at = grip0 + [0, 0, TABLE_PINCH_Z]
        _, R = self.gripper_frame()
        self.p = self.point_world(TABLE_PINCH)
        self.log.setdefault("table_pick_at", []).append(np.round(np.r_[grip0, np.degrees(np.arctan2(hdir[1], hdir[0]))], 3).tolist())
        self.play(Segment("to_dropped_peeler", 2.0, at + [0, 0, TABLE_ABOVE], OPEN,
                          targets=self.pinch_targets(blend(R[:, 2], down), blend(HEAD_SIDE * R[:, 1], hdir),
                                                     TABLE_PINCH)), on_frame)
        tg = self.pinch_targets(lambda s: down, lambda s: hdir, TABLE_PINCH)
        self.play(Segment("descend_dropped_peeler", 1.5, at, targets=tg), on_frame)
        _, R = self.gripper_frame()
        self.log.setdefault("table_pick_fit", []).append(dict(
            pinch_err_mm=np.round((self.point_world(TABLE_PINCH) - at) * 1000, 1).tolist(),
            y_dot_h=round(float(HEAD_SIDE * R[:, 1] @ hdir), 3), z_up=round(float(R[:, 2] @ down), 3),
            q_deg=np.round(np.degrees(self.q_arm), 1).tolist()))
        self.play(Segment("close_dropped_peeler", 1.0, grip_to=CLOSED_PEELER, targets=tg), on_frame)
        z0 = t.d.xpos[t.peeler_b][2]
        self.play(Segment("lift_dropped_peeler", 1.2, at + [0, 0, PEELER_LIFT], targets=tg), on_frame)
        rise = t.d.xpos[t.peeler_b][2] - z0
        self.log.setdefault("table_pick_rise_mm", []).append(round(rise * 1000, 1))
        if rise < MIN_PEELER_RISE:
            raise GraspFailed(f"dropped peeler rose only {rise * 1000:.0f} mm")

    def set_down_at_spot(self, on_frame):
        """Lay the peeler (held straight down, picked from the table) on the table at PEELER_SPOT with its handle
        tangential and the head where the usual tilted grasp wants it. Rotating a straight-down hand to the tilted
        grasp would roll the peeler 50 deg about its handle, so it is set down and taken again instead."""
        t = self.task
        th = np.arctan2(PEELER_SPOT[1], PEELER_SPOT[0])
        hdir = -np.array([-np.sin(th), np.cos(th), 0.0])         # head towards -tangent, like on the pegs
        down = np.array([0, 0, 1.0])
        _, R = self.gripper_frame()
        self.p = self.point_world(TABLE_PINCH)
        offset = lambda: self.site("peeler_grip") - self.point_world(TABLE_PINCH)
        spot = np.r_[PEELER_SPOT, 0.006 + SPOT_CLEAR]
        tg_move = self.pinch_targets(lambda s: down, blend(HEAD_SIDE * R[:, 1], hdir), TABLE_PINCH)
        self.play(Segment("to_spot", 2.0, lambda: spot + [0, 0, TABLE_ABOVE] - offset(), targets=tg_move), on_frame)
        tg = self.pinch_targets(lambda s: down, lambda s: hdir, TABLE_PINCH)
        self.play(Segment("lower_at_spot", 1.2, lambda: spot - offset(), targets=tg), on_frame)
        self.play(Segment("release_at_spot", 0.6, grip_to=OPEN, targets=tg), on_frame)
        self.play(Segment("leave_spot", 0.8, self.p + [0, 0, TABLE_ABOVE], targets=tg), on_frame)
        self.log.setdefault("spot_pose", []).append(np.round(np.r_[self.site("peeler_grip"), self.peeler_y()], 3).tolist())

    def recover_peeler(self, on_frame):
        """The peeler is out of the hand: go home, wait until it lies still, pick it up from the table, put it back
        on the pegs, and take it again with the usual grasp."""
        self.log.setdefault("peeler_drops", []).append(np.round(self.site("peeler_grip"), 3).tolist())
        self.play(Segment("peeler_drop_home", 1.5, grip_to=OPEN, joint_to=self.home_q()), on_frame)
        self.p = self.point_world(PINCH)
        for _ in range(int(SETTLE_MAX / 0.2)):
            if self.peeler_speed() < SETTLE_SPEED:
                break
            self.play(Segment("peeler_drop_wait", 0.2, joint_to=np.r_[self.q_arm]), on_frame)
        self.pick_peeler_from_table(on_frame)
        self.set_down_at_spot(on_frame)
        self.pick_peeler(on_frame, TABLE_PINCH, "spot_", SPOT_TILT)   # tilted like the usual grasp -> the usual in-hand roll
        self.put_peeler(on_frame)          # back on the pegs in the usual pose, then the usual grasp
        self.pick_peeler(on_frame)

    # ---------- peeling ----------
    def peel_face(self, on_frame):
        t = self.task
        r = t.apple_r
        self.p = t.blade_pos()
        psi0 = STROKE_PSI[0]
        start = self.surface(LAT_TOP + 5, r + STANDOFF, psi0)
        self.play(Segment("to_apple", 2.5, start, targets=self.blade_targets(*self.blend_from_current(psi0, start))),
                  on_frame)
        # hold away from the skin with the stroke's own targets so the arm settles before the blade goes in
        self.play(Segment("align", 0.6, targets=self.blade_targets(lambda s: self.blade_dir(psi0))), on_frame)
        lat_cur, psi_cur = LAT_TOP + 5, psi0
        for psi in STROKE_PSI:
            if psi != psi_cur or lat_cur != LAT_TOP + 5:
                # around the apple at the stand-off radius to the top of this stroke
                path = lambda s, a=(lat_cur, psi_cur), b=(LAT_TOP + 5, psi): self.surface(
                    a[0] + s * (b[0] - a[0]), r + STANDOFF, a[1] + s * (b[1] - a[1]))
                bd = lambda s, a=psi_cur, b=psi: self.blade_dir(a + s * (b - a))
                self.play(Segment("reposition", 1.0, path=path, targets=self.blade_targets(bd)), on_frame)
            tg = self.blade_targets(lambda s, psi=psi: self.blade_dir(psi))
            self.play(Segment("engage", 0.5, path=lambda s, psi=psi: self.surface(
                LAT_TOP + 5 * (1 - s), r + STANDOFF + s * (SKIN_CLEAR - STANDOFF), psi), targets=tg), on_frame)
            self.play(Segment("stroke", 1.8, path=lambda s, psi=psi: self.surface(
                LAT_TOP + s * (LAT_BOTTOM - LAT_TOP), r + SKIN_CLEAR, psi), targets=tg), on_frame)
            self.play(Segment("disengage", 0.5, path=lambda s, psi=psi: self.surface(
                LAT_BOTTOM, r + SKIN_CLEAR + s * (STANDOFF - SKIN_CLEAR), psi), targets=tg), on_frame)
            lat_cur, psi_cur = LAT_BOTTOM, psi
        # back away from the apple, outwards and up
        tg = self.blade_targets(lambda s: self.blade_dir(psi_cur))
        self.play(Segment("back_off", 1.0, path=lambda s: self.surface(
            LAT_BOTTOM + s * (40 - LAT_BOTTOM), r + STANDOFF + s * 0.04, psi_cur), targets=tg), on_frame)
        # straight up before heading to the stand: a direct move sweeps the handle through the apple
        self.play(Segment("raise", 0.8, self.p + [0, 0, RAISE], targets=tg), on_frame)
        self.log.setdefault("peeled_after_face", []).append(round(t.peeled_fraction(), 3))

    # ---------- turning the apple ----------
    def twist_cycle(self, on_frame, amount):
        """One regrasp: approach along the gripper axis with the jaws at TWIST_FROM, close, turn the jaw line by
        `amount` deg (the apple follows), open, back off along the gripper axis."""
        c = self.task.apple_pos()
        ax = gripper_axis(c, TWIST_TILT)
        _, R = self.gripper_frame()
        a_now = self.current_jaw_angle()
        self.p = self.point_world(GRASP_LOCAL)
        self.play(Segment("twist_approach", 1.5, c + APPROACH_D * ax, OPEN,
                          targets=self.twist_targets(lambda s: a_now + s * (TWIST_FROM - a_now), blend(R[:, 2], ax))),
                  on_frame)
        hold = self.twist_targets(lambda s: TWIST_FROM, lambda s: ax)
        self.play(Segment("twist_descend", 1.0, c, targets=hold), on_frame)
        self.play(Segment("twist_close", 0.6, grip_to=CLOSED, targets=hold), on_frame)
        a_end = TWIST_FROM + amount
        self.play(Segment("twist", max(0.4, 1.2 * amount / TWIST_MAX),
                          targets=self.twist_targets(lambda s: TWIST_FROM + s * amount)), on_frame)
        free = self.twist_targets(lambda s: a_end)
        self.play(Segment("twist_open", 0.6, grip_to=OPEN, targets=free), on_frame)
        z1 = self.gripper_frame()[1][:, 2]
        self.play(Segment("twist_retreat", 0.8, self.p + APPROACH_D * z1,
                          targets=self.twist_targets(lambda s: a_end, lambda s: z1)), on_frame)

    def turn_apple_to(self, on_frame, target_rad):
        """Regrasp-and-turn until the turntable is within TURN_TOL of target (privileged angle feedback)."""
        t = self.task
        for _ in range(MAX_TWISTS):
            remaining = np.degrees(target_rad - t.turntable_angle())
            if remaining < TURN_TOL + TURN_BIAS:
                break
            self.twist_cycle(on_frame, min(TWIST_MAX, remaining - TURN_BIAS))
        self.log.setdefault("turn_error_deg", []).append(round(np.degrees(t.turntable_angle() - target_rad), 1))

    def run(self, on_frame):
        self.start()
        self.skewer(on_frame)
        base = self.task.turntable_angle()
        for face in range(N_FACES):
            self.pick_peeler(on_frame)
            if self.peeler_drop is not None and self.peeler_drop[0] == face:
                self.drop_peeler(on_frame, self.peeler_drop[1])
                self.recover_peeler(on_frame)
            for _ in range(MAX_PEELER_DROPS):
                if self.peeler_held():
                    break
                self.recover_peeler(on_frame)
            self.peel_face(on_frame)
            self.put_peeler(on_frame)
            if face < N_FACES - 1:
                self.turn_apple_to(on_frame, base + np.radians(90.0) * (face + 1))
        self.play(Segment("home", 2.0, joint_to=self.home_q()), on_frame)
