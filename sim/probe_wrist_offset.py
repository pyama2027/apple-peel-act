"""Render the wrist camera with increasing forward offsets to find where it clears its own housing mesh."""
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/models/so101/apple_scene.xml"))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

cid = m.camera("wrist").id
base = m.cam_pos[cid].copy()
R = np.zeros(9)
mujoco.mju_quat2Mat(R, m.cam_quat[cid])
fwd = -R.reshape(3, 3)[:, 2]  # camera looks along its -z axis
r = mujoco.Renderer(m, 224, 224)
tiles = []
for off in (0.0, 0.01, 0.02, 0.03, 0.04):
    m.cam_pos[cid] = base + off * fwd
    r.update_scene(d, camera="wrist")
    img = r.render()
    dark = (img.max(axis=2) < 30).mean()
    print(f"offset={off:.2f} m  pos={np.round(m.cam_pos[cid], 4)}  dark_frac={dark:.2f}")
    tiles.append(img)
r.close()
Image.fromarray(np.concatenate(tiles, axis=1)).save(ROOT / "outputs/day1_scene/wrist_offsets.png")
