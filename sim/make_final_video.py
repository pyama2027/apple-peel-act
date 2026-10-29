"""Speed up an evaluation video for the final submission (user: the inference video must fit in 3 minutes; speed up
the video, not the robot) and stamp the speed factor in the corner.

Usage: python make_final_video.py in.mp4 out.mp4 [--speed auto|1.5] [--max-sec 180] [--until-sec T]
  --speed auto picks the smallest factor (in steps of 0.1) that brings the clip (up to --until-sec) under --max-sec.
"""
import argparse
import math

import av
import numpy as np
from PIL import Image, ImageDraw

from task import FPS
from video import VideoWriter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--speed", default="auto")
    p.add_argument("--max-sec", type=float, default=180.0)
    p.add_argument("--until-sec", type=float, default=None, help="cut the source here (e.g. when the task is done)")
    args = p.parse_args()

    with av.open(args.src) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    if args.until_sec is not None:
        frames = frames[:int(args.until_sec * FPS)]
    dur = len(frames) / FPS
    speed = (max(1.0, math.ceil(dur / args.max_sec * 10) / 10) if args.speed == "auto" else float(args.speed))
    h, w = frames[0].shape[:2]
    vw = VideoWriter(args.dst, FPS, w, h)
    n_out = int(len(frames) / speed)
    label = f"x{speed:g}"
    for i in range(n_out):
        im = Image.fromarray(frames[min(len(frames) - 1, int(round(i * speed)))])
        dr = ImageDraw.Draw(im)
        dr.rectangle((w - 52, h - 22, w - 4, h - 4), fill=(0, 0, 0))
        dr.text((w - 46, h - 19), label, fill=(255, 255, 0))
        vw.write(np.asarray(im))
    vw.close()
    print(f"{args.src}: {dur:.0f} s -> {args.dst}: {n_out / FPS:.0f} s at {label}")


if __name__ == "__main__":
    main()
