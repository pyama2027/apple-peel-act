"""Merge the per-process shards written by record_dataset.py into one LeRobot dataset.

Usage: python merge_shards.py --task peel   (data/so101_peel_sim_shard* -> data/so101_peel_sim)
"""
import argparse
import shutil

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from task import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="peel")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    name = f"so101_{args.task}_sim"
    shards = sorted((ROOT / "data").glob(f"{name}_shard*"), key=lambda d: int(d.name.rsplit("shard", 1)[1]))
    out = ROOT / "data" / name
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists (use --overwrite)")
        shutil.rmtree(out)
    aggregate_datasets([f"local/{d.name}" for d in shards], f"local/{name}", roots=shards, aggr_root=out)
    logs = "".join((d / "record_log.jsonl").read_text() for d in shards if (d / "record_log.jsonl").exists())
    (out / "record_log.jsonl").write_text(logs)
    ds = LeRobotDataset(f"local/{name}", root=out)
    print(f"merged {len(shards)} shards -> {out}: {ds.num_episodes} episodes, {ds.num_frames} frames")


if __name__ == "__main__":
    main()
