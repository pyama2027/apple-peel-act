"""Damped least-squares IK for SO101 on multiple point targets.

A 5-DoF arm cannot hit an arbitrary 6-D pose, so orientation is expressed as a second point target:
e.g. tip at P and the gripper body origin at P + L * up gives a top-down approach.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


@dataclass
class PointTarget:
    kind: str  # "site", "body", or "point" (fixed to body `name` at body-frame offset `local`, e.g. a held tool)
    name: str
    pos: np.ndarray
    weight: float = 1.0
    local: np.ndarray | None = None


def _point(model, data, t):
    if t.kind == "point":
        i = model.body(t.name).id
        p = data.xpos[i] + data.xmat[i].reshape(3, 3) @ t.local
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, None, p, i)
        return p, jacp
    if t.kind == "site":
        i = model.site(t.name).id
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, i)
        return data.site_xpos[i], jacp
    i = model.body(t.name).id
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, None, i)
    return data.xpos[i], jacp


def solve_ik(model, targets, q_init, iters=200, damping=1e-3, tol=1e-4, joints=ARM_JOINTS):
    """Return (qpos for arm joints, final max position error [m]). Uses a scratch MjData; never steps physics."""
    data = mujoco.MjData(model)
    jids = [model.joint(j).id for j in joints]
    qadr = np.array([model.jnt_qposadr[j] for j in jids])
    dadr = np.array([model.jnt_dofadr[j] for j in jids])
    lo, hi = model.jnt_range[jids, 0], model.jnt_range[jids, 1]
    data.qpos[qadr] = np.clip(q_init, lo, hi)

    err_max = np.inf
    for _ in range(iters):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        errs, jacs = [], []
        for t in targets:
            p, J = _point(model, data, t)
            errs.append(t.weight * (t.pos - p))
            jacs.append(t.weight * J[:, dadr])
        e, J = np.concatenate(errs), np.concatenate(jacs)
        err_max = np.abs(e).max()
        if err_max < tol:
            break
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(len(e)), e)
        data.qpos[qadr] = np.clip(data.qpos[qadr] + dq, lo, hi)
    return data.qpos[qadr].copy(), float(err_max)


def gripper_axis(p, tilt):
    """Unit vector from the grasp point back up the gripper (z axis of the gripper frame) for a target at p.

    tilt [rad]: lean the gripper axis away from vertical, top towards the robot base (the wrist cannot flex far
    enough to point straight down from high poses). tilt=0 -> vertical gripper.
    """
    radial = np.array([p[0], p[1], 0.0])
    radial /= max(np.linalg.norm(radial), 1e-9)
    return np.cos(tilt) * np.array([0, 0, 1.0]) - np.sin(tilt) * radial


def grasp_targets(p, up=0.07, tilt=0.0):
    """'grasp' site at p and 'grasp_axis' site (placed `up` along the gripper z axis) behind it."""
    p = np.asarray(p, dtype=float)
    return [PointTarget("site", "grasp", p),
            PointTarget("site", "grasp_axis", p + up * gripper_axis(p, tilt), weight=0.5)]


def top_down_targets(tip_pos, height=0.098):  # 0.098 = |gripperframe - gripper origin| in the MJCF
    """Tip at tip_pos, gripper body origin straight above it -> jaws point down."""
    tip_pos = np.asarray(tip_pos, dtype=float)
    return [PointTarget("site", "gripperframe", tip_pos),
            PointTarget("body", "gripper", tip_pos + np.array([0, 0, height]), weight=0.5)]
