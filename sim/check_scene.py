"""Day1 check: load the apple/needle scene, let it settle, render overhead + wrist + an overview image."""
import argparse
import os
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "sim/models/so101/apple_scene.xml"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--out", default=str(ROOT / "outputs/day1_scene"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    # arm at mid-range ("new_calib" zero), gripper open
    d.ctrl[:] = m.actuator_ctrlrange.mean(axis=1)
    d.ctrl[m.actuator("gripper").id] = m.actuator_ctrlrange[m.actuator("gripper").id, 1]
    apple = m.body("apple").id
    z0 = m.body_pos[apple][2]
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    print(f"apple start z={z0:.4f}  settled pos={np.round(d.xpos[apple], 4)}  |v|={np.linalg.norm(d.cvel[apple]):.2e}")
    print(f"contacts={d.ncon}")

    r = mujoco.Renderer(m, args.size, args.size)
    tiles = []
    for cam in ("overhead", "wrist"):
        r.update_scene(d, camera=cam)
        img = r.render()
        Image.fromarray(img).save(f"{args.out}/{cam}.png")
        tiles.append(img)
    free = mujoco.MjvCamera()
    free.lookat[:] = [0.15, 0, 0.05]
    free.distance, free.azimuth, free.elevation = 0.9, 200, -30
    r.update_scene(d, camera=free)
    tiles.append(r.render())
    Image.fromarray(np.concatenate(tiles, axis=1)).save(f"{args.out}/scene_tiles.png")
    print("saved", args.out)


if __name__ == "__main__":
    main()
