"""Run one scripted demo episode and write a review video + metrics (no dataset output).

Usage: python run_demo.py --seed 0 [--apple-x 0.2 --apple-y -0.15]
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from expert import Expert, GraspFailed
from expert_peel import PeelExpert
from task import FPS, ROOT, AppleTask, PeelTask
from video import VideoWriter

# y > -0.12 (apple nearly in front of the base) failed 4/4 grasps in the first sweep -> excluded
APPLE_X, APPLE_Y = (0.15, 0.25), (-0.20, -0.12)


DROP_PROB, DROP_FRAC, PEELER_DROP_FRAC = 0.3, (0.15, 0.35), (0.55, 0.7)   # share of episodes where the expert drops the apple on purpose during the
                                        # transfer (and picks it up again), and how far into the transfer


NEEDLE_XY0, NEEDLE_JITTER = np.array([0.20, 0.15]), 0.01   # needle stand placed within +-1 cm
PEELER_JITTER = (0.008, 0.01, np.radians(8))               # peeler on the pegs: +-dx, +-dy [m], +-yaw [rad]
ARM_JITTER = np.radians(8)                                  # start posture: home +- this per arm joint


def sample_scene(rng):
    """Apple placement first (same draws as before), then the 2026-10-28 diversity: needle, peeler, start posture."""
    scene = dict(apple_xy=[rng.uniform(*APPLE_X), rng.uniform(*APPLE_Y)], apple_yaw=rng.uniform(-np.pi, np.pi))
    scene["needle_xy"] = (NEEDLE_XY0 + rng.uniform(-NEEDLE_JITTER, NEEDLE_JITTER, 2)).tolist()
    scene["peeler_offset"] = [rng.uniform(-j, j) for j in PEELER_JITTER]
    scene["arm_dq"] = np.r_[rng.uniform(-ARM_JITTER, ARM_JITTER, 5), rng.uniform(-0.2, 0.6)].tolist()
    return scene


def reset_task(task, home_q, scene):
    kw = dict(apple_xy=scene["apple_xy"], apple_yaw=scene["apple_yaw"], needle_xy=scene.get("needle_xy"),
              q_arm=np.asarray(home_q) + np.asarray(scene.get("arm_dq", np.zeros(6))))
    if isinstance(task, PeelTask):
        kw["peeler_offset"] = scene.get("peeler_offset")
    task.reset(**kw)


def sample_peeler_drop(seed):
    """Per-seed peeler drop plan: (face, fraction of the way to the apple) or None."""
    rng = np.random.default_rng([seed, 8])
    return (int(rng.integers(4)), rng.uniform(*PEELER_DROP_FRAC)) if rng.random() < DROP_PROB else None


MISS_PROB, MISS_DIST, MAX_MISSES = 0.3, (0.007, 0.012), 3   # planned skewer misses: chance per insert, aim offset


def sample_miss(seed):
    """Per-seed list of off-centre aims for the first inserts: each insert misses with MISS_PROB (at most MAX_MISSES)."""
    rng = np.random.default_rng([seed, 9])
    plan = []
    while len(plan) < MAX_MISSES and rng.random() < MISS_PROB:
        a, r = rng.uniform(-np.pi, np.pi), rng.uniform(*MISS_DIST)
        plan.append([r * np.cos(a), r * np.sin(a)])
    return plan


def sample_drop(seed):
    """Per-seed drop plan (own RNG stream, so the apple placements of old seeds are unchanged): fraction or None."""
    rng = np.random.default_rng([seed, 7])
    return rng.uniform(*DROP_FRAC) if rng.random() < DROP_PROB else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--apple-x", type=float)
    p.add_argument("--apple-y", type=float)
    p.add_argument("--out", default=str(ROOT / "outputs/day2_demo"))
    p.add_argument("--no-video", action="store_true", help="metrics only (fast)")
    p.add_argument("--task", choices=("skewer", "peel"), default="skewer")
    p.add_argument("--drop", type=float, help="force a drop this far into the transfer (default: per-seed plan)")
    p.add_argument("--miss", type=float, help="force one skewer miss with this x offset [m]")
    p.add_argument("--no-jitter", action="store_true", help="no needle / peeler / start-posture diversity")
    p.add_argument("--peeler-drop", type=float, nargs=2, metavar=("FACE", "FRAC"), help="force a peeler drop")
    p.add_argument("--view", choices=("wide", "closeup"), default="wide",
                   help="review camera: whole table, or close on the apple from the side being peeled")
    p.add_argument("--frames", type=int, nargs=2, metavar=("FROM", "TO"),
                   help="render only frames [FROM, TO) to demo_seed<N>_<FROM>.mp4 (for parallel rendering)")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    scene = sample_scene(rng)
    if args.apple_x is not None:
        scene["apple_xy"] = [args.apple_x, args.apple_y]
    if args.no_jitter:
        scene = {k: scene[k] for k in ("apple_xy", "apple_yaw")}

    peel = args.task == "peel"
    task = PeelTask() if peel else AppleTask()
    expert = PeelExpert(task) if peel else Expert(task)
    reset_task(task, expert.home_q(), scene)
    expert.drop_frac = args.drop if args.drop is not None else sample_drop(args.seed)
    expert.miss_plan = [[args.miss, 0.0]] if args.miss is not None else sample_miss(args.seed)
    n_planned_miss = len(expert.miss_plan)
    if peel:
        expert.peeler_drop = ((int(args.peeler_drop[0]), args.peeler_drop[1]) if args.peeler_drop
                              else sample_peeler_drop(args.seed))
    m, d = task.m, task.d

    robot_bodies = {i for i in range(m.nbody) if i not in (0, task.apple_b, task.needle_b)
                    and not (peel and i == task.peeler_b)}
    table = m.geom("table").id
    stats = dict(max_pen_apple_robot=0.0, max_pen_robot_table=0.0, frames=0)

    free = mujoco.MjvCamera()
    if args.view == "closeup":
        free.lookat[:] = [0.21, 0.10, 0.10]
        free.distance, free.azimuth, free.elevation = 0.42, 300, -22
    else:
        free.lookat[:] = [0.2, 0.0, 0.08]
        free.distance, free.azimuth, free.elevation = 0.75, 215, -25
    if not args.no_video:
        big = mujoco.Renderer(m, 336, 336)
        name = f"demo_seed{args.seed}_{args.frames[0]:05d}.mp4" if args.frames else f"demo_seed{args.seed}.mp4"
        vw = VideoWriter(out / name, FPS, 336 + 2 * 224, 336)

    def on_frame(seg, action_q):
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            pen = max(0.0, -c.dist)
            if task.apple_g in (c.geom1, c.geom2) and ({b1, b2} - {task.apple_b}) <= robot_bodies:
                stats["max_pen_apple_robot"] = max(stats["max_pen_apple_robot"], pen)
            if table in (c.geom1, c.geom2) and ({b1, b2} & robot_bodies):
                stats["max_pen_robot_table"] = max(stats["max_pen_robot_table"], pen)
        stats["frames"] += 1
        if args.no_video or (args.frames and not args.frames[0] <= stats["frames"] - 1 < args.frames[1]):
            return
        imgs = task.render()
        big.update_scene(d, camera=free)
        canvas = Image.new("RGB", (336 + 2 * 224, 336), (20, 20, 20))
        canvas.paste(Image.fromarray(big.render()), (0, 0))
        canvas.paste(Image.fromarray(imgs["overhead"]), (336, 0))
        canvas.paste(Image.fromarray(imgs["wrist"]), (336 + 224, 0))
        dr = ImageDraw.Draw(canvas)
        dr.text((6, 6), f"seed {args.seed}  {seg}  t={stats['frames'] / FPS:4.1f}s  skewered={task.skewered}", fill=(255, 255, 255))
        if peel:
            dr.text((6, 20), f"peeled {task.peeled_fraction() * 100:5.1f}%  turntable {np.degrees(task.turntable_angle()):6.1f} deg",
                    fill=(255, 255, 255))
        dr.text((342, 232), "overhead 224x224", fill=(255, 255, 255))
        dr.text((342 + 224, 232), "wrist 224x224", fill=(255, 255, 255))
        vw.write(np.asarray(canvas))

    aborted = None
    try:
        expert.run(on_frame)
    except GraspFailed as e:
        aborted = str(e)
    if not args.no_video:
        vw.close()
        big.close()
    task.close()

    res = dict(seed=args.seed, planned_misses=n_planned_miss, drop_frac=expert.drop_frac, peeler_drop=getattr(expert, 'peeler_drop', None), scene={k: np.round(v, 4).tolist() for k, v in scene.items()},
               success=aborted is None and bool(task.success()), skewered=bool(task.skewered), aborted=aborted,
               final_depth_cm=round(task.insertion_depth() * 100, 2),
               **(dict(peeled=round(task.peeled_fraction(), 3)) if peel else {}),
               duration_s=round(stats["frames"] / FPS, 2), frames=stats["frames"],
               max_pen_apple_robot_mm=round(stats["max_pen_apple_robot"] * 1000, 2),
               max_pen_robot_table_mm=round(stats["max_pen_robot_table"] * 1000, 2),
               max_ik_err_mm=round(expert.log.get("max_ik_err", 0) * 1000, 2),
               **{k: v for k, v in expert.log.items() if k not in ("max_ik_err", "insert_trace")})
    trace = expert.log.get("insert_trace", [])
    res["insert_trace_dxy_depth_mm"] = trace[::6] + trace[-1:]
    print(json.dumps(res, ensure_ascii=False))
    (out / f"demo_seed{args.seed}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
