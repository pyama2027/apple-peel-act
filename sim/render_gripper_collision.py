"""Explainer figure: gripper visual mesh vs. fixed-jaw collision shape before (mesh convex hull) and after (boxes)."""
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ORIG = ROOT / "third_party/SO-ARM100/Simulation/SO101/so101_new_calib_camera.xml"
OURS = ROOT / "sim/models/so101/apple_scene.xml"
W, H = 480, 480
BLUE = np.array([0.1, 0.45, 1.0, 0.55])
GREEN = np.array([0.1, 0.8, 0.3, 0.55])


def render(xml, show_bodies_collision, title, apple_r=0.04):
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    d.qpos[m.jnt_qposadr[m.joint("gripper").id]] = 1.0
    mujoco.mj_forward(m, d)
    g = m.body("gripper").id
    Rg, pg = d.xmat[g].reshape(3, 3), d.xpos[g]

    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [0, 0, 1, 0, 0, 0]  # visual meshes only
    if show_bodies_collision:
        opt.geomgroup[3] = 1
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 1  # draw mesh collision geoms as MuJoCo sees them
        for i in range(m.ngeom):
            if m.geom_group[i] != 3:
                continue
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i])
            is_mesh = m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
            mesh = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[i]) if is_mesh else ""
            # fixed jaw = the wrist_roll_follower mesh (before) or our primitive boxes (after); hide the motor
            if body in show_bodies_collision and (not is_mesh or mesh == "wrist_roll_follower_so101_v1"):
                m.geom_rgba[i] = BLUE
                m.geom_matid[i] = -1
            elif body == "moving_jaw_so101_v1":
                m.geom_rgba[i] = GREEN
                m.geom_matid[i] = -1
            else:
                m.geom_rgba[i, 3] = 0

    cam = mujoco.MjvCamera()
    cam.lookat[:] = pg + Rg @ np.array([0.01, 0, -0.05])
    view = Rg[:, 1]  # look along the gripper's y axis: jaws open in the image plane
    cam.azimuth = np.degrees(np.arctan2(view[1], view[0]))
    cam.elevation = np.degrees(np.arcsin(-view[2]))
    cam.distance = 0.22

    r = mujoco.Renderer(m, H, W)
    r.update_scene(d, camera=cam, scene_option=opt)
    if apple_r:  # outline of an 8 cm apple at the grasp position, for scale
        c = pg + Rg @ np.array([-0.010 + apple_r + 0.002, 0, -0.065])
        s = r.scene
        if s.ngeom < s.maxgeom:
            mujoco.mjv_initGeom(s.geoms[s.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [apple_r, 0, 0], c,
                                np.eye(3).ravel(), np.array([0.85, 0.1, 0.1, 0.25], dtype=np.float32))
            s.ngeom += 1
    img = Image.fromarray(r.render())
    r.close()
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, W, 26], fill=(255, 255, 255))
    dr.text((8, 7), title, fill=(0, 0, 0))
    return img


panels = [
    render(OURS, None, "1) visual mesh (what you see)  + red ghost = 8cm apple"),
    render(ORIG, {"gripper"}, "2) BEFORE: fixed-jaw collision = convex hull (blue)"),
    render(OURS, {"gripper"}, "3) AFTER: palm box + finger box (blue)  moving jaw (green)"),
]
out = ROOT / "outputs/day2_grasp/gripper_collision_before_after.png"
Image.fromarray(np.concatenate([np.asarray(p) for p in panels], axis=1)).save(out)
print("saved", out)
