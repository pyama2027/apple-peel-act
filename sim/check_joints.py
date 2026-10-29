"""Day1 check: load SO101 MJCF, drive each joint through its range, log tracking error, save snapshots.

Usage: python sim/check_joints.py [--xml PATH] [--out DIR]
"""
import argparse
import os
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = ROOT / "third_party/SO-ARM100/Simulation/SO101/scene.xml"


def settle(model, data, ctrl, seconds=1.5):
    data.ctrl[:] = ctrl
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default=str(DEFAULT_XML))
    p.add_argument("--out", default=str(ROOT / "outputs/day1_joints"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    print(f"timestep={model.opt.timestep}  nq={model.nq}  nu={model.nu}")

    renderer = mujoco.Renderer(model, 360, 480)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, -0.15, 0.15]
    cam.distance, cam.azimuth, cam.elevation = 0.8, 135, -25

    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    home = (lo + hi) / 2
    settle(model, data, home)

    frames = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        qadr = model.jnt_qposadr[model.actuator_trnid[i, 0]]
        for frac in (0.8, 0.2):
            target = home.copy()
            target[i] = lo[i] + frac * (hi[i] - lo[i])
            settle(model, data, target)
            err = data.qpos[qadr] - target[i]
            print(f"{name:14s} target={target[i]:+.3f} rad  actual={data.qpos[qadr]:+.3f}  err={err:+.4f}")
            renderer.update_scene(data, cam)
            img = renderer.render()
            Image.fromarray(img).save(f"{args.out}/{i}_{name}_{int(frac * 100)}.png")
            frames.append(img)
        settle(model, data, home)

    # contact sheet for quick review
    rows = [np.concatenate(frames[k:k + 4], axis=1) for k in range(0, len(frames), 4)]
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    Image.fromarray(np.concatenate(rows, axis=0)).save(f"{args.out}/contact_sheet.png")
    print("saved", args.out)


if __name__ == "__main__":
    main()
