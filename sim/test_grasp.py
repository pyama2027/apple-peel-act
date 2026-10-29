"""Physics grasp test: for several apple radii, place the open gripper around the apple, close, lift 5 cm, hold.
Success = apple rises with the gripper and stays there."""
import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from ik import ARM_JOINTS, grasp_targets, solve_ik

ROOT = Path(__file__).resolve().parents[1]
OPEN, CLOSED = 1.0, -0.17  # gripper joint [rad]


def run(m, radius, out_dir, renderer, OPEN=OPEN):
    d = mujoco.MjData(m)
    apple_g, apple_b = m.geom("apple").id, m.body("apple").id
    m.geom_size[apple_g, 0] = radius
    m.body_pos[apple_b, 2] = radius
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    center = d.xpos[apple_b].copy()

    arm = [m.actuator(j).id for j in ARM_JOINTS]
    qadr = [m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS]
    g_act, g_q = m.actuator("gripper").id, m.jnt_qposadr[m.joint("gripper").id]

    q_grasp, e1 = solve_ik(m, grasp_targets(center), np.zeros(5))
    q_lift, e2 = solve_ik(m, grasp_targets(center + [0, 0, 0.05]), q_grasp)
    d.qpos[qadr], d.qpos[g_q] = q_grasp, OPEN
    d.ctrl[arm], d.ctrl[g_act] = q_grasp, OPEN
    mujoco.mj_forward(m, d)

    frames = []
    def step(seconds, arm_from=None, arm_to=None, grip_from=None, grip_to=None):
        n = int(seconds / m.opt.timestep)
        for k in range(n):
            a = (k + 1) / n
            if arm_to is not None:
                d.ctrl[arm] = (1 - a) * arm_from + a * arm_to
            if grip_to is not None:
                d.ctrl[g_act] = (1 - a) * grip_from + a * grip_to
            mujoco.mj_step(m, d)
        renderer.update_scene(d, camera="wrist")
        w = renderer.render()
        renderer.update_scene(d, camera="overhead")
        frames.append(np.concatenate([renderer.render(), w], axis=1))

    step(0.3)
    step(1.0, grip_from=OPEN, grip_to=CLOSED)
    grip_closed = d.qpos[g_q]
    site = m.site("grasp").id
    rel = lambda: d.xpos[apple_b] - d.site_xpos[site]  # apple center relative to the grasp site
    z_before, site_z_before, rel_before = d.xpos[apple_b][2], d.site_xpos[site][2], rel()
    step(1.5, arm_from=q_grasp, arm_to=q_lift)
    rel_lifted = rel()
    step(1.0)
    rise = d.xpos[apple_b][2] - z_before
    site_rise = d.site_xpos[site][2] - site_z_before
    sag = np.linalg.norm(rel_lifted - rel_before)  # settling while the load comes on
    slip = np.linalg.norm(rel() - rel_lifted)  # creep during the static hold: this is what must be ~0
    print(f"   sag during lift={sag * 1000:.1f}mm  (rel z {rel_before[2] * 1000:+.1f} -> {rel_lifted[2] * 1000:+.1f}mm)")
    ok = rise > 0.03 and slip < 0.002
    Image.fromarray(np.concatenate(frames, axis=0)).save(out_dir / f"grasp_r{int(radius * 1000)}mm.png")
    print(f"r={radius * 100:.1f}cm ik_err={e1 * 1000:.1f}/{e2 * 1000:.1f}mm gripper_closed_at={grip_closed:+.3f}rad "
          f"hand_rise={site_rise * 100:.1f}cm apple_rise={rise * 100:.1f}cm slip={slip * 1000:.1f}mm "
          f"-> {'OK' if ok else 'FAIL'}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--radii", type=float, nargs="+", default=[0.030, 0.035, 0.040])
    p.add_argument("--noslip", type=int, default=None, help="override option.noslip_iterations")
    p.add_argument("--friction", type=float, default=None, help="override sliding friction of apple + jaw geoms")
    p.add_argument("--open", type=float, default=OPEN, help="gripper opening before the grasp [rad]")
    args = p.parse_args()
    out = ROOT / "outputs/day2_grasp"
    out.mkdir(parents=True, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/models/so101/apple_scene.xml"))
    if args.noslip is not None:
        m.opt.noslip_iterations = args.noslip
    if args.friction is not None:
        jaw = m.body("moving_jaw_so101_v1").id
        for g in [m.geom("apple").id, m.geom("fixed_finger_col").id, *np.where(m.geom_bodyid == jaw)[0]]:
            m.geom_friction[g, 0] = args.friction
    print(f"noslip_iterations={m.opt.noslip_iterations} apple_friction={m.geom_friction[m.geom('apple').id]}")
    r = mujoco.Renderer(m, 224, 224)
    for radius in args.radii:
        run(m, radius, out, r, OPEN=args.open)
    r.close()


if __name__ == "__main__":
    main()
