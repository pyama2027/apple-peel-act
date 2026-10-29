"""Roll out a trained LeRobot policy (e.g. ACT) on the apple skewer + peel task in the sim.

Each control frame the policy gets exactly what the dataset recorded (observation.state in LeRobot units and the
overhead / wrist 224x224 images) and its action is applied as the joint-position target. Apple placements come from
seeds outside the training range (dataset: 3000-3052).

Usage: python eval_policy.py --policy outputs/train/act_peel/checkpoints/last/pretrained_model --seeds 5000 5001
       [--max-frames 6600] [--video] [--out outputs/eval/act_peel]
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from PIL import Image, ImageDraw

from expert import Expert
from record_dataset import TASK_TEXT
from run_demo import reset_task, sample_scene
from task import CAMERAS, FPS, ROOT, PeelTask
from video import VideoWriter


def to_batch(task, device):
    """Sim state -> policy observation dict (batch of 1), same layout as the dataset frames."""
    obs = {"observation.state": torch.from_numpy(task.q_to_lerobot(task.qpos())).float()[None].to(device)}
    imgs = task.render()
    for cam in CAMERAS:
        obs[f"observation.images.{cam}"] = (torch.from_numpy(imgs[cam]).permute(2, 0, 1).float() / 255.0)[None].to(device)
    obs["task"] = [TASK_TEXT["peel"]]
    return obs, imgs


def rollout(policy, pre, post, seed, max_frames, video_path=None):
    task = PeelTask()
    reset_task(task, Expert(task).home_q(), sample_scene(np.random.default_rng(seed)))
    policy.reset()
    device = next(policy.parameters()).device
    log = dict(seed=seed, skewered_frame=None, peeled_90_frame=None, success_frame=None)
    if video_path:
        big = mujoco.Renderer(task.m, 336, 336)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.21, 0.10, 0.10]
        cam.distance, cam.azimuth, cam.elevation = 0.42, 300, -22
        vw = VideoWriter(video_path, FPS, 336 + 2 * 224, 336)
    for k in range(max_frames):
        obs, imgs = to_batch(task, device)
        with torch.inference_mode():
            action = post(policy.select_action(pre(obs)))
        q = task.lerobot_to_q(action[0].cpu().numpy())
        if video_path:
            big.update_scene(task.d, camera=cam)
            canvas = Image.new("RGB", (336 + 2 * 224, 336), (20, 20, 20))
            canvas.paste(Image.fromarray(big.render()), (0, 0))
            for i, c in enumerate(CAMERAS):
                canvas.paste(Image.fromarray(imgs[c]), (336 + i * 224, 0))
            dr = ImageDraw.Draw(canvas)
            dr.text((6, 6), f"POLICY  seed {seed}  t={k / FPS:5.1f}s  skewered={task.skewered}", fill=(255, 255, 0))
            dr.text((6, 20), f"peeled {task.peeled_fraction() * 100:5.1f}%  turntable {np.degrees(task.turntable_angle()):6.1f} deg",
                    fill=(255, 255, 0))
            vw.write(np.asarray(canvas))
        task.step(q)
        if task.skewered and log["skewered_frame"] is None:
            log["skewered_frame"] = k
        if task.peeled_fraction() >= 0.9 and log["peeled_90_frame"] is None:
            log["peeled_90_frame"] = k
        if log["success_frame"] is None and task.success():
            log["success_frame"] = k
    if video_path:
        vw.close()
        big.close()
    log.update(frames=max_frames, skewered=bool(task.skewered), peeled=round(task.peeled_fraction(), 3),
               success_at_end=bool(task.success()), turntable_deg=round(float(np.degrees(task.turntable_angle())), 1),
               apple_xy=np.round(sample_scene(np.random.default_rng(seed))["apple_xy"], 4).tolist())
    task.close()
    return log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="path to a pretrained_model directory")
    p.add_argument("--seeds", type=int, nargs="+", default=[5000])
    p.add_argument("--max-frames", type=int, default=6600, help="dataset episodes are ~6100 frames")
    p.add_argument("--video", action="store_true")
    p.add_argument("--out", default=str(ROOT / "outputs/eval/act_peel"))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = ACTPolicy.from_pretrained(args.policy)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.policy,
                                         preprocessor_overrides={"device_processor": {"device": device}})
    for seed in args.seeds:
        log = rollout(policy, pre, post, seed, args.max_frames, out / f"eval_seed{seed}.mp4" if args.video else None)
        log["policy"] = str(args.policy)
        print(json.dumps(log), flush=True)
        (out / f"eval_seed{seed}.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
