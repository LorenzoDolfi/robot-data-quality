"""Shared helpers: locate a LeRobot dataset on disk and load per-episode arrays from its parquet files."""
import glob
import os

import numpy as np
import pandas as pd


def dataset_root(repo_id=None, root=None):
    if root:
        return os.path.abspath(root)
    try:  # LeRobot >= 0.4
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    return str(LeRobotDataset(repo_id).root)


def data_files(root):
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files under {root}/data")
    return files


def load_episodes(root):
    """Return {episode_index: {"action": (T,2), "state": (T,2), "reward": (T,)}} ordered by frame."""
    cols = ["episode_index", "frame_index", "action", "observation.state", "next.reward"]
    frames = []
    for f in data_files(root):
        df = pd.read_parquet(f)
        frames.append(df[[c for c in cols if c in df.columns]])
    df = pd.concat(frames, ignore_index=True)
    eps = {}
    for ep, g in df.groupby("episode_index", sort=True):
        g = g.sort_values("frame_index")
        eps[int(ep)] = {
            "action": np.stack(g["action"].to_numpy()).astype(np.float64),
            "state": np.stack(g["observation.state"].to_numpy()).astype(np.float64),
            "reward": g["next.reward"].to_numpy(dtype=float) if "next.reward" in g else np.zeros(len(g)),
        }
    return eps
