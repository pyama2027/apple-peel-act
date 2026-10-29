"""Identify what occludes the wrist camera by hiding candidate geoms one at a time; also report the nearest geom on the optical axis."""
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/models/so101/apple_scene.xml"))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

cid = m.camera("wrist").id
pos = d.cam_xpos[cid]
fwd = -d.cam_xmat[cid].reshape(3, 3)[:, 2]
gid = np.array([-1], dtype=np.int32)
dist = mujoco.mj_ray(m, d, pos, fwd, None, 1, -1, gid)
hit = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid[0]) or f"geom#{gid[0]}"
body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid[0]]) if gid[0] >= 0 else None
print(f"cam world pos={np.round(pos, 4)} fwd={np.round(fwd, 3)}")
print(f"optical-axis ray hits: {hit} (body={body}) at {dist:.4f} m")

r = mujoco.Renderer(m, 224, 224)
tiles = []
orig = m.geom_rgba.copy()
for label, bodies in (("none", []), ("wrist_camera", ["wrist_camera"]),
                      ("cam_mount", ["wrist_camera_mount"]), ("needle", ["needle"])):
    m.geom_rgba[:] = orig
    for b in bodies:
        m.geom_rgba[m.geom_bodyid == m.body(b).id, 3] = 0
    r.update_scene(d, camera="wrist")
    img = r.render()
    print(f"hide {label:12s} dark_frac={(img.max(axis=2) < 30).mean():.2f}")
    tiles.append(img)
r.close()
Image.fromarray(np.concatenate(tiles, axis=1)).save(ROOT / "outputs/day1_scene/wrist_occluder.png")
