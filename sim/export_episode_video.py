"""Export one episode of a recorded LeRobot dataset as a review video: overhead | wrist side by side, exactly the
frames the policy is trained on (decoded from the dataset's own video files), with the recorded state overlaid.

Usage: python export_episode_video.py --episode 0 [--root ../data/so101_peel_sim] [--out ../outputs/dataset_ep0.mp4]
"""
import argparse
from pathlib import Path

import av
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image, ImageDraw

from task import CAMERAS, FPS, ROOT
from video import VideoWriter

SCALE = 2  # 224 -> 448 per camera for viewing


def decode(path, t0, t1):
    frames = []
    with av.open(str(path)) as c:
        for f in c.decode(video=0):
            t = float(f.pts * f.time_base)
            if t < t0 - 0.5 / FPS:
                continue
            if t > t1 - 0.5 / FPS:
                break
            frames.append(f.to_ndarray(format="rgb24"))
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--root", default=str(ROOT / "data/so101_peel_sim"))
    p.add_argument("--out", default=None)
    args = p.parse_args()
    root = Path(args.root)
    ds = LeRobotDataset(f"local/{root.name}", root=root)
    e = ds.meta.episodes[args.episode]
    cams = {}
    for cam in CAMERAS:
        key = f"observation.images.{cam}"
        path = root / ds.meta.video_path.format(video_key=key, chunk_index=e[f"videos/{key}/chunk_index"],
                                                 file_index=e[f"videos/{key}/file_index"])
        cams[cam] = decode(path, e[f"videos/{key}/from_timestamp"], e[f"videos/{key}/to_timestamp"])
    n = min(len(v) for v in cams.values())
    start = e["dataset_from_index"]
    states = ds.hf_dataset.select(range(start, start + n))["observation.state"]
    names = ds.features["observation.state"]["names"]

    out = Path(args.out or ROOT / f"outputs/dataset_ep{args.episode}.mp4")
    size = 224 * SCALE
    vw = VideoWriter(out, FPS, 2 * size, size + 40)
    for i in range(n):
        canvas = Image.new("RGB", (2 * size, size + 40), (20, 20, 20))
        for k, cam in enumerate(CAMERAS):
            canvas.paste(Image.fromarray(cams[cam][i]).resize((size, size), Image.NEAREST), (k * size, 40))
        dr = ImageDraw.Draw(canvas)
        dr.text((6, 4), f"dataset episode {args.episode}  frame {i}/{n}  t={i / FPS:5.1f}s   overhead | wrist (224x224 as stored)",
                fill=(255, 255, 255))
        st = np.asarray(states[i])
        dr.text((6, 20), "state  " + "  ".join(f"{nm.split('.')[0]} {v:6.1f}" for nm, v in zip(names, st)),
                fill=(200, 200, 200))
        vw.write(np.asarray(canvas))
    vw.close()
    print(f"wrote {out}: {n} frames ({n / FPS:.0f} s), episode length {e['length']}")


if __name__ == "__main__":
    main()
