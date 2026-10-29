"""Find the front face of the wrist_camera housing along the camera's optical axis and propose a camera pos just in front of it."""
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/models/so101/apple_scene.xml"))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

cid = m.camera("wrist").id
cam_pos = d.cam_xpos[cid]
fwd = -d.cam_xmat[cid].reshape(3, 3)[:, 2]

body = m.body("wrist_camera").id
verts = []
for g in np.where(m.geom_bodyid == body)[0]:
    if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        continue
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    verts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
v = np.concatenate(verts)
print("housing extent (world, m):", np.round(v.max(0) - v.min(0), 4))
along = (v - cam_pos) @ fwd
lateral = np.linalg.norm((v - cam_pos) - np.outer(along, fwd), axis=1)
print(f"along-axis range: {along.min():+.4f} .. {along.max():+.4f} m")
# front face: max projection among vertices close to the optical axis (lens barrel)
near = lateral < 0.006
front = along[near].max() if near.any() else along.max()
print(f"front of housing near axis: {front:+.4f} m (vertices near axis: {near.sum()})")

off = front + 0.002
new_world = cam_pos + off * fwd
g = m.body("gripper").id
new_local = d.xmat[g].reshape(3, 3).T @ (new_world - d.xpos[g])
print(f"offset along axis: {off:.4f} m -> camera pos in gripper frame: {' '.join(f'{x:.4f}' for x in new_local)}")
