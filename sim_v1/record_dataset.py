"""Record scripted-expert episodes into a LeRobot dataset (SO101 follower format, 30 fps, overhead + wrist cameras).

Frames are streamed into the dataset's episode buffer (images go to disk), and an episode is saved only if the expert
succeeds; failed / aborted episodes are discarded with clear_episode_buffer().

Rendering is the bottleneck (software GL under WSL), so record with several processes, each writing its own shard
(--shard i --num-shards k takes seeds seed0+i, seed0+i+k, ...), then merge them with merge_shards.py.

Usage: python record_dataset.py --task peel --episodes 10 --shard 0 --num-shards 5 [--seed0 3000]
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

from expert import Expert, GraspFailed
from expert_peel import PeelExpert
from run_demo import sample_scene
from task import CAMERAS, FPS, IMG_SIZE, ROOT, STATE_NAMES, AppleTask, PeelTask

TASK_TEXT = {
    "skewer": "Pick up the apple and skewer it onto the needle.",
    "peel": "Skewer the apple onto the needle, then peel it with the peeler, turning the apple by hand.",
}


def features():
    hw = {n: float for n in STATE_NAMES}
    obs = {**hw, **{c: (IMG_SIZE, IMG_SIZE, 3) for c in CAMERAS}}
    return {**hw_to_dataset_features(hw, "action"), **hw_to_dataset_features(obs, "observation")}


def run_episode(task_name, seed, ds, ft):
    """Stream one expert episode into ds's episode buffer. Returns info; the caller saves or discards the buffer."""
    rng = np.random.default_rng(seed)
    scene = sample_scene(rng)
    task = PeelTask() if task_name == "peel" else AppleTask()
    expert = PeelExpert(task) if task_name == "peel" else Expert(task)
    task.reset(**scene, q_arm=expert.home_q())
    n = [0]

    def on_frame(seg, action_q):
        obs = {**dict(zip(STATE_NAMES, task.q_to_lerobot(task.qpos()))), **task.render()}
        act = dict(zip(STATE_NAMES, task.q_to_lerobot(action_q)))
        ds.add_frame({**build_dataset_frame(ft, obs, "observation"), **build_dataset_frame(ft, act, "action"),
                      "task": TASK_TEXT[task_name]})
        n[0] += 1

    aborted = None
    try:
        expert.run(on_frame)
    except GraspFailed as e:
        aborted = str(e)
    ok = aborted is None and bool(task.success())
    info = dict(seed=seed, apple_xy=np.round(scene["apple_xy"], 4).tolist(), success=ok, aborted=aborted, frames=n[0])
    if task_name == "peel":
        info["peeled"] = round(task.peeled_fraction(), 3)
    task.close()
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=("skewer", "peel"), default="peel")
    p.add_argument("--episodes", type=int, default=50, help="number of successful episodes to record (this shard)")
    p.add_argument("--seed0", type=int, default=3000)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-tries", type=int, default=None)
    p.add_argument("--root", default=None, help="default: data/so101_<task>_sim[_shard<i>]")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    name = f"so101_{args.task}_sim" + (f"_shard{args.shard}" if args.num_shards > 1 else "")
    root = Path(args.root) if args.root else ROOT / "data" / name
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} exists (use --overwrite)")
        shutil.rmtree(root)

    ft = features()
    ds = LeRobotDataset.create(f"local/{name}", FPS, ft, root=root, robot_type="so101_follower", use_videos=True,
                               image_writer_threads=4)
    log = []
    max_tries = args.max_tries or 2 * args.episodes
    seed = args.seed0 + args.shard
    try:
        while sum(r["success"] for r in log) < args.episodes and len(log) < max_tries:
            info = run_episode(args.task, seed, ds, ft)
            if info["success"]:
                ds.save_episode()
            else:
                ds.clear_episode_buffer()
            log.append(info)
            print(json.dumps(info), flush=True)
            seed += args.num_shards
    finally:
        ds.finalize()
        (root / "record_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
    print(f"recorded {sum(r['success'] for r in log)} episodes from {len(log)} tries -> {root}")


if __name__ == "__main__":
    main()
