"""Grasp feasibility check v2 (honest version).

For each apple radius:
  1. search a grasp point in the gripper frame where the apple is clear of every robot collision geom
     (>= CLEAR m) but as deep between the fingers as possible,
  2. IK the arm there (jaws open), close, lift 5 cm, hold 1 s,
  3. report max penetration, sag, slip and save side-view images.
PASS = max penetration <= 2 mm, hold slip <= 2 mm, apple rose >= 3 cm.
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np

from ik import ARM_JOINTS, solve_ik
from inspect_grasp import BLUE, GREEN, side_view
from test_grasp import CLOSED, grasp_targets
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLEAR = 0.003
TABLE_MARGIN = 0.008  # fingertip height above the table at the grasp pose


def robot_collision_geoms(m):
    robot_bodies = {m.body(n).id for n in ("gripper", "moving_jaw_so101_v1", "wrist", "lower_arm")}
    return [i for i in range(m.ngeom) if m.geom_bodyid[i] in robot_bodies and (m.geom_contype[i] or m.geom_conaffinity[i])]


def min_dist(m, d, apple, geoms):
    fromto = np.zeros(6)
    return min(mujoco.mj_geomDistance(m, d, apple, g, 0.05, fromto) for g in geoms)


def find_grasp_point(m, radius, open_q, geoms):
    """Grid-search apple center (gripper frame, y=0) that clears all robot geoms; prefer deepest (largest z)."""
    d = mujoco.MjData(m)
    d.qpos[m.jnt_qposadr[m.joint("gripper").id]] = open_q
    mujoco.mj_forward(m, d)
    g = m.body("gripper").id
    Rg, pg = d.xmat[g].reshape(3, 3).copy(), d.xpos[g].copy()
    apple_q = m.jnt_qposadr[m.joint("apple_free").id]
    apple = m.geom("apple").id
    best = None
    # fingertips must stay above the table, which is at the apple's bottom (z_c - r in the gripper frame)
    tip_z = min(m.geom_pos[i][2] - m.geom_size[i][2] for i in (m.geom("fixed_finger_ext_col").id,))
    z_start = min(-0.03 - radius, tip_z + radius - TABLE_MARGIN)
    for z in np.arange(z_start, -0.16, -0.002):                  # deeper first (closer to the palm)
        for x in np.arange(-0.010 + radius, 0.06, 0.002):       # from the fixed finger outward
            d.qpos[apple_q:apple_q + 3] = pg + Rg @ np.array([x, 0, z])
            mujoco.mj_forward(m, d)
            dist = min_dist(m, d, apple, geoms)
            if dist >= CLEAR:
                best = (x, z, dist)
                break
        if best:
            return np.array([best[0], 0, best[1]]), best[2]
    return None, None


def run(m, radius, open_q, solimp, out_dir, r):
    apple_g, apple_b = m.geom("apple").id, m.body("apple").id
    m.geom_size[apple_g, 0] = radius
    m.body_pos[apple_b, 2] = radius
    if solimp:
        m.geom_solimp[apple_g, :3] = solimp  # (dmin, dmax, width); keep midpoint/power
    geoms = robot_collision_geoms(m)
    local, clear = find_grasp_point(m, radius, open_q, geoms)
    if local is None:
        print(f"r={radius * 100:.1f}cm: no collision-free grasp point found")
        return
    m.site_pos[m.site("grasp").id] = local
    m.site_pos[m.site("grasp_axis").id] = local + [0, 0, 0.07]

    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    center = d.xpos[apple_b].copy()
    arm = [m.actuator(j).id for j in ARM_JOINTS]
    qadr = [m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS]
    g_act, g_q = m.actuator("gripper").id, m.jnt_qposadr[m.joint("gripper").id]
    q_grasp, e1 = solve_ik(m, grasp_targets(center), np.zeros(5))
    q_lift, _ = solve_ik(m, grasp_targets(center + [0, 0, 0.05]), q_grasp)
    d.qpos[qadr], d.qpos[g_q] = q_grasp, open_q
    d.ctrl[arm], d.ctrl[g_act] = q_grasp, open_q
    mujoco.mj_forward(m, d)

    worst = [0.0]
    def pen():
        v = [-c.dist for c in d.contact[:d.ncon] if apple_g in (c.geom1, c.geom2) and
             ({c.geom1, c.geom2} - {apple_g}).pop() in geoms]
        return max(v, default=0.0)

    def go(seconds, arm_to=None, grip_to=None):
        a0, g0 = d.ctrl[arm].copy(), d.ctrl[g_act]
        n = int(seconds / m.opt.timestep)
        for k in range(n):
            a = (k + 1) / n
            if arm_to is not None:
                d.ctrl[arm] = (1 - a) * a0 + a * arm_to
            if grip_to is not None:
                d.ctrl[g_act] = (1 - a) * g0 + a * grip_to
            mujoco.mj_step(m, d)
            worst[0] = max(worst[0], pen())

    tiles = []
    site = m.site("grasp").id
    rel = lambda: d.xpos[apple_b] - d.site_xpos[site]
    pen0 = pen()
    go(0.3)
    tiles.append(side_view(m, d, r, f"r={radius * 100:.1f}cm open"))
    go(1.0, grip_to=CLOSED)
    tiles.append(side_view(m, d, r, f"closed  pen={pen() * 1000:.1f}mm"))
    z0, rel0 = d.xpos[apple_b][2], rel()
    go(1.5, arm_to=q_lift)
    rel1 = rel()
    go(1.0)
    tiles.append(side_view(m, d, r, f"lift+hold  pen={pen() * 1000:.1f}mm"))
    m.geom_rgba[apple_g], m.geom_matid[apple_g] = [0.75, 0.08, 0.08, 1], m.mat("apple").id
    tiles.append(side_view(m, d, r, "lift+hold (visual only)", show_collision=False))
    rise, slip, sag = d.xpos[apple_b][2] - z0, np.linalg.norm(rel() - rel1), np.linalg.norm(rel1 - rel0)
    ok = worst[0] <= 0.002 and slip <= 0.002 and rise >= 0.03
    print(f"r={radius * 100:.1f}cm grasp_pt(gripper frame)=({local[0] * 1000:+.0f},{local[2] * 1000:+.0f})mm "
          f"clear={clear * 1000:.1f}mm pen@start={pen0 * 1000:.1f}mm closed_at={d.qpos[g_q]:+.2f}rad "
          f"max_pen={worst[0] * 1000:.1f}mm rise={rise * 100:.1f}cm sag={sag * 1000:.1f}mm slip={slip * 1000:.1f}mm "
          f"-> {'PASS' if ok else 'FAIL'}")
    Image.fromarray(np.concatenate(tiles, axis=1)).save(out_dir / f"check_r{int(radius * 1000)}mm.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--radii", type=float, nargs="+", default=[0.025, 0.030, 0.035, 0.040])
    p.add_argument("--open", type=float, default=1.5)
    p.add_argument("--solimp", type=float, nargs=3, default=[0.95, 0.99, 0.001])
    p.add_argument("--xml", default=str(ROOT / "sim/models/so101/apple_scene.xml"))
    args = p.parse_args()
    out = ROOT / "outputs/day2_grasp"
    out.mkdir(parents=True, exist_ok=True)
    for radius in args.radii:
        m = mujoco.MjModel.from_xml_path(args.xml)
        jaw = m.body("moving_jaw_so101_v1").id
        for i in range(m.ngeom):  # colour collision shapes for the side view
            if m.geom_group[i] == 3:
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
                if m.geom_bodyid[i] == jaw:
                    m.geom_rgba[i], m.geom_matid[i] = GREEN, -1
                elif nm.endswith("_col"):
                    m.geom_rgba[i] = BLUE
                else:
                    m.geom_rgba[i, 3] = 0
        m.geom_rgba[m.geom("apple").id] = [0.9, 0.1, 0.1, 0.6]
        m.geom_matid[m.geom("apple").id] = -1
        r = mujoco.Renderer(m, 300, 300)
        run(m, radius, args.open, args.solimp, out, r)
        r.close()


if __name__ == "__main__":
    main()
