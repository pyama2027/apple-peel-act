"""Inspect the real physics state of the grasp test: per-contact penetration between apple and robot,
and side-view renders (visual + collision shapes) of the actual simulated apple at each phase."""
import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from ik import ARM_JOINTS, solve_ik
from test_grasp import CLOSED, grasp_targets

ROOT = Path(__file__).resolve().parents[1]
BLUE, GREEN = [0.1, 0.45, 1.0, 0.5], [0.1, 0.8, 0.3, 0.5]


def side_view(m, d, r, title, show_collision=True):
    g = m.body("gripper").id
    Rg, pg = d.xmat[g].reshape(3, 3), d.xpos[g]
    cam = mujoco.MjvCamera()
    cam.lookat[:] = pg + Rg @ np.array([0.01, 0, -0.05])
    v = Rg[:, 1]
    cam.azimuth, cam.elevation, cam.distance = np.degrees(np.arctan2(v[1], v[0])), np.degrees(np.arcsin(-v[2])), 0.25
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 0, 1, int(show_collision), 0, 0]
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = int(show_collision)
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(show_collision)
    r.update_scene(d, camera=cam, scene_option=opt)
    img = Image.fromarray(r.render())
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, img.width, 20], fill=(255, 255, 255))
    dr.text((6, 4), title, fill=(0, 0, 0))
    return np.asarray(img)


def apple_contacts(m, d):
    apple = m.geom("apple").id
    rows = []
    for c in d.contact[:d.ncon]:
        if apple in (c.geom1, c.geom2):
            o = c.geom2 if c.geom1 == apple else c.geom1
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, o) or (
                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[o]) + "(mesh)")
            rows.append((nm, c.dist))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--radius", type=float, default=0.04)
    p.add_argument("--open", type=float, default=1.5)
    args = p.parse_args()

    m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/models/so101/apple_scene.xml"))
    # colour collision shapes: fixed-jaw boxes blue, moving jaw green, other robot collision meshes hidden
    jaw = m.body("moving_jaw_so101_v1").id
    for i in range(m.ngeom):
        if m.geom_group[i] == 3:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if name in ("palm_col", "fixed_finger_col"):
                m.geom_rgba[i] = BLUE
            elif m.geom_bodyid[i] == jaw:
                m.geom_rgba[i], m.geom_matid[i] = GREEN, -1
            else:
                m.geom_rgba[i, 3] = 0
    d = mujoco.MjData(m)
    apple_g, apple_b = m.geom("apple").id, m.body("apple").id
    m.geom_size[apple_g, 0] = args.radius
    m.body_pos[apple_b, 2] = args.radius
    m.geom_rgba[apple_g] = [0.9, 0.1, 0.1, 0.6]
    m.geom_matid[apple_g] = -1
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    center = d.xpos[apple_b].copy()

    arm = [m.actuator(j).id for j in ARM_JOINTS]
    qadr = [m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS]
    g_act, g_q = m.actuator("gripper").id, m.jnt_qposadr[m.joint("gripper").id]
    q_grasp, _ = solve_ik(m, grasp_targets(center), np.zeros(5))
    q_lift, _ = solve_ik(m, grasp_targets(center + [0, 0, 0.05]), q_grasp)
    d.qpos[qadr], d.qpos[g_q] = q_grasp, args.open
    d.ctrl[arm], d.ctrl[g_act] = q_grasp, args.open
    mujoco.mj_forward(m, d)

    r = mujoco.Renderer(m, 360, 360)
    tiles = []

    def report(phase):
        rows = apple_contacts(m, d)
        worst = min([dist for _, dist in rows], default=0.0)
        print(f"[{phase:14s}] gripper={d.qpos[g_q]:+.3f}rad  apple contacts: "
              + (", ".join(f"{n} {dist * 1000:+.1f}mm" for n, dist in rows) or "none")
              + f"   -> worst penetration {max(0, -worst) * 1000:.1f}mm")
        tiles.append(side_view(m, d, r, f"{phase}: worst penetration {max(0, -worst) * 1000:.1f} mm"))

    report("placed (t=0)")

    def run(seconds, arm_to=None, grip_to=None):
        a0, g0 = d.ctrl[arm].copy(), d.ctrl[g_act]
        n = int(seconds / m.opt.timestep)
        for k in range(n):
            a = (k + 1) / n
            if arm_to is not None:
                d.ctrl[arm] = (1 - a) * a0 + a * arm_to
            if grip_to is not None:
                d.ctrl[g_act] = (1 - a) * g0 + a * grip_to
            mujoco.mj_step(m, d)

    run(0.3)
    report("settled")
    run(1.0, grip_to=CLOSED)
    report("closed")
    run(1.5, arm_to=q_lift)
    run(1.0)
    report("lifted+hold")
    r.close()
    out = ROOT / f"outputs/day2_grasp/inspect_r{int(args.radius * 1000)}mm_open{args.open:.1f}.png"
    Image.fromarray(np.concatenate(tiles, axis=1)).save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
