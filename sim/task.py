"""Apple-on-needle task core: model loading, LeRobot unit conversion, cameras, skewer (weld) logic, success check.

Units follow LeRobot's SO101 follower (use_degrees=True): 5 arm joints in degrees, gripper in 0..100 (0 = closed).
"""
import os
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "sim/models/so101/apple_scene.xml"

FPS = 30
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
STATE_NAMES = [f"{j}.pos" for j in JOINTS]
CAMERAS = ("overhead", "wrist")
IMG_SIZE = 224

SKEWER_XY_TOL = 0.005    # apple centre within 5 mm of the needle axis
SKEWER_DEPTH = 0.020     # needle at least 2 cm inside the apple


class AppleTask:
    def __init__(self, xml=SCENE, joints=JOINTS):
        self.joints = joints
        self.state_names = [f"{j}.pos" for j in joints]
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.n_sub = round(1.0 / FPS / m.opt.timestep)
        assert abs(self.n_sub * m.opt.timestep - 1.0 / FPS) < 1e-9, "timestep must divide 1/FPS"
        self.act = np.array([m.actuator(j).id for j in joints])
        self.qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in joints])
        self.g_lo, self.g_hi = m.jnt_range[m.joint("gripper").id]
        self.apple_b = m.body("apple").id
        self.apple_g = m.geom("apple").id
        self.apple_r = float(m.geom_size[self.apple_g, 0])
        self.apple_q = m.jnt_qposadr[m.joint("apple_free").id]
        self.needle_b = m.body("needle").id
        self.needle_tip = m.site("needle_tip").id
        self.weld = m.equality("apple_on_needle").id
        self.needle_g = m.geom("needle_shaft").id
        self.needle_aff = int(m.geom_conaffinity[self.needle_g])
        self.renderer = None
        self.skewered = False
        mujoco.mj_forward(m, self.d)  # so site/body positions are valid before the first reset()

    # ---------- units ----------
    def q_to_lerobot(self, q):
        """sim joint vector (rad) -> LeRobot units (degrees, except gripper 0..100)."""
        q = np.asarray(q, dtype=float)
        out = np.degrees(q).copy()
        out[5] = 100.0 * (q[5] - self.g_lo) / (self.g_hi - self.g_lo)
        return out.astype(np.float32)

    def lerobot_to_q(self, v):
        v = np.asarray(v, dtype=float)
        q = np.radians(v).copy()
        q[5] = self.g_lo + v[5] / 100.0 * (self.g_hi - self.g_lo)
        return q

    # ---------- episode ----------
    def reset(self, apple_xy, apple_yaw=0.0, needle_xy=None, q_arm=None):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        if needle_xy is not None:
            m.body_pos[self.needle_b, :2] = needle_xy
        d.qpos[self.apple_q:self.apple_q + 3] = [apple_xy[0], apple_xy[1], self.apple_r]
        d.qpos[self.apple_q + 3:self.apple_q + 7] = [np.cos(apple_yaw / 2), 0, 0, np.sin(apple_yaw / 2)]
        d.eq_active[self.weld] = 0
        m.geom_conaffinity[self.needle_g] = self.needle_aff
        if q_arm is not None:  # may omit trailing joints (e.g. the turntable), which then start at 0
            d.qpos[self.qadr[:len(q_arm)]] = q_arm
            d.ctrl[self.act[:len(q_arm)]] = q_arm
        mujoco.mj_forward(m, d)
        self.skewered = False

    def qpos(self):
        return self.d.qpos[self.qadr].copy()

    def step(self, ctrl_q):
        """Apply a joint-position target (rad, one per task joint) for one control frame (1/FPS s)."""
        self.d.ctrl[self.act] = ctrl_q
        for _ in range(self.n_sub):
            self.update_needle_contact()
            mujoco.mj_step(self.m, self.d)
        if not self.skewered and self.skewer_ready():
            self.activate_weld()

    # ---------- skewer ----------
    def apple_pos(self):
        return self.d.xpos[self.apple_b].copy()

    def needle_tip_pos(self):
        return self.d.site_xpos[self.needle_tip].copy()

    def insertion_depth(self):
        """How far the needle tip is inside the apple, measured from the apple's bottom (m)."""
        return self.needle_tip_pos()[2] - (self.apple_pos()[2] - self.apple_r)

    def update_needle_contact(self):
        """Before the skewer, the needle and the apple only touch when the apple comes down off-centre: then it rests
        on the needle tip (a visible miss) instead of sliding through it. A centred apple slides on (the weld)."""
        if self.skewered:
            aff = self.needle_aff
        elif self.insertion_depth() > 0.002:
            return                                   # already on the needle: keep the state it came down with
        else:
            dxy = np.linalg.norm(self.apple_pos()[:2] - self.needle_tip_pos()[:2])
            aff = self.needle_aff | (2 if dxy >= SKEWER_XY_TOL else 0)   # 2 = the apple's contype
        self.m.geom_conaffinity[self.needle_g] = aff

    def skewer_ready(self):
        dxy = np.linalg.norm(self.apple_pos()[:2] - self.needle_tip_pos()[:2])
        return dxy < SKEWER_XY_TOL and self.insertion_depth() > SKEWER_DEPTH

    def activate_weld(self):
        """Weld apple to needle at the current relative pose (relpose = body2 pose expressed in body1 frame,
        verified by sim/check_weld.py: no jump on activation)."""
        m, d = self.m, self.d
        b1, b2 = self.needle_b, self.apple_b
        R1 = d.xmat[b1].reshape(3, 3)
        relpos = R1.T @ (d.xpos[b2] - d.xpos[b1])
        q1inv, relq = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(q1inv, d.xquat[b1])
        mujoco.mju_mulQuat(relq, q1inv, d.xquat[b2])
        m.eq_data[self.weld, 0:3] = 0           # anchor at body2 origin
        m.eq_data[self.weld, 3:6] = relpos
        m.eq_data[self.weld, 6:10] = relq
        d.eq_active[self.weld] = 1
        self.skewered = True

    def success(self):
        """Apple skewered and no longer held by the gripper (resting on the needle by itself)."""
        if not self.skewered:
            return False
        jaw_bodies = {self.m.body("gripper").id, self.m.body("moving_jaw_so101_v1").id}
        for c in self.d.contact[:self.d.ncon]:
            if self.apple_g in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.apple_g else c.geom1
                if self.m.geom_bodyid[other] in jaw_bodies:
                    return False
        return True

    # ---------- rendering ----------
    def render(self, size=IMG_SIZE):
        if self.renderer is None or self.renderer.width != size:
            self.renderer = mujoco.Renderer(self.m, size, size)
        out = {}
        for cam in CAMERAS:
            self.renderer.update_scene(self.d, camera=cam)
            out[cam] = self.renderer.render().copy()
        return out

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


# ---------------------------------------------------------------------------------------------------------------
PEEL_SCENE = Path(os.environ.get("PEEL_SCENE", ROOT / "sim/models/so101/peel_scene.xml"))   # generated by sim/build_peel_scene.py
PEEL_TOL = 0.004       # a point of the blade edge within 4 mm (radially) of the apple surface cuts
BLADE_SAMPLES = 9      # points along the blade edge checked every frame
PATCH_REACH = 0.006    # a cutting blade point removes skin patches whose centre is within this distance
PEEL_DONE = 0.90       # success needs this fraction of the peelable skin removed


class PeelTask(AppleTask):
    """Apple-on-needle + peeling. Skin patches (visual geoms) are hidden where the peeler's blade edge lies on the
    apple surface: only the blade cuts, not the side of the head or the handle."""

    def __init__(self, xml=PEEL_SCENE):
        super().__init__(xml)
        m = self.m
        self.turn_adr = m.jnt_qposadr[m.joint("turntable").id]
        self.skin = np.array([i for i in range(m.ngeom) if m.geom(i).name.startswith("skin_")])
        pos = m.geom_pos[self.skin]
        self.skin_dir = pos / np.linalg.norm(pos, axis=1, keepdims=True)  # unit vectors in the apple frame
        self.skin_rgba = m.geom_rgba[self.skin].copy()
        self.blade = m.site("peeler_blade").id
        self.blade_g = m.geom("peeler_blade").id
        self.peeler_b = m.body("peeler").id
        self.peeled = np.zeros(len(self.skin), bool)

    def reset(self, *args, peeler_offset=None, **kw):
        """peeler_offset = (dx, dy, dyaw [rad]) of the peeler on its pegs (scene diversity)."""
        self.m.geom_rgba[self.skin] = self.skin_rgba
        self.peeled[:] = False
        super().reset(*args, **kw)
        if peeler_offset is not None:
            m, d = self.m, self.d
            adr = m.jnt_qposadr[m.joint("peeler_free").id]
            d.qpos[adr:adr + 2] += peeler_offset[:2]
            dq = np.array([np.cos(peeler_offset[2] / 2), 0, 0, np.sin(peeler_offset[2] / 2)])
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, dq, d.qpos[adr + 3:adr + 7].copy())
            d.qpos[adr + 3:adr + 7] = q
            mujoco.mj_forward(m, d)

    def step(self, ctrl_q):
        super().step(ctrl_q)
        if self.skewered:
            self.peel_update()

    def blade_pos(self):
        return self.d.site_xpos[self.blade].copy()

    def blade_edge(self):
        """Points along the blade edge (the capsule axis), world frame, shape (BLADE_SAMPLES, 3)."""
        d, g = self.d, self.blade_g
        axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
        half = self.m.geom_size[g, 1]
        return d.geom_xpos[g] + np.linspace(-half, half, BLADE_SAMPLES)[:, None] * axis

    def peel_update(self):
        v = (self.blade_edge() - self.apple_pos()) @ self.d.xmat[self.apple_b].reshape(3, 3)  # apple frame
        dist = np.linalg.norm(v, axis=1)
        cutting = np.abs(dist - self.apple_r) < PEEL_TOL
        if not cutting.any():
            return
        on_skin = v[cutting] / dist[cutting, None]
        gap = np.linalg.norm(self.skin_dir[:, None, :] - on_skin[None, :, :], axis=2).min(axis=1) * self.apple_r
        new = (gap < PATCH_REACH) & ~self.peeled
        if new.any():
            self.m.geom_rgba[self.skin[new], 3] = 0.0
            self.peeled |= new

    def turntable_angle(self):
        return float(self.d.qpos[self.turn_adr])

    def peeled_fraction(self):
        return float(self.peeled.mean())

    def success(self):
        return super().success() and self.peeled_fraction() >= PEEL_DONE
