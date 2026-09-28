"""
Winnow trial 1: score every episode in a LeRobot dataset and build training subsets.

Outputs (in --out_dir):
  episode_scores.csv  one row per episode with its metrics and quality score
  subsets.json        episode index lists: full, curated, random, bottom
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

try:  # LeRobot >= 0.4
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:  # older layout
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def load_frames(repo_id):
    ds = LeRobotDataset(repo_id)
    wanted = ["episode_index", "action", "next.reward", "next.success"]
    cols = [c for c in wanted if c in ds.hf_dataset.column_names]
    df = ds.hf_dataset.select_columns(cols).to_pandas()
    return df


def score_episode(g):
    actions = np.stack(g["action"].to_numpy()).astype(np.float64)
    n = len(actions)
    reward = g["next.reward"].to_numpy(dtype=float) if "next.reward" in g else np.zeros(n)
    success = bool(g["next.success"].any()) if "next.success" in g else False

    step = np.linalg.norm(np.diff(actions, axis=0), axis=1) if n > 1 else np.zeros(1)
    scale = np.median(step) + 1e-9
    idle_frac = float(np.mean(step < 0.1 * scale))
    jerk = np.linalg.norm(np.diff(actions, n=2, axis=0), axis=1) if n > 2 else np.zeros(1)
    jerk_ratio = float(np.mean(jerk) / scale)

    return {
        "length": n,
        "max_reward": float(reward.max()) if n else 0.0,
        "final_reward": float(reward[-1]) if n else 0.0,
        "success": success,
        "idle_frac": idle_frac,
        "jerk_ratio": jerk_ratio,
    }


def minmax(x):
    x = np.asarray(x, dtype=float)
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", default="lerobot/pusht")
    p.add_argument("--keep_frac", type=float, default=0.6, help="fraction of episodes kept in the curated subset")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="trial_outputs")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    df = load_frames(a.repo_id)
    rows = []
    for ep, g in df.groupby("episode_index", sort=True):
        r = score_episode(g)
        r["episode_index"] = int(ep)
        rows.append(r)
    s = pd.DataFrame(rows)

    # Higher is better. Weights are a first guess; tune them as you learn what matters.
    s["quality"] = (
        0.45 * s["max_reward"].clip(0, 1)
        + 0.15 * s["final_reward"].clip(0, 1)
        + 0.15 * (1 - minmax(s["length"]))
        + 0.10 * (1 - minmax(s["idle_frac"]))
        + 0.15 * (1 - minmax(s["jerk_ratio"]))
    )
    s = s.sort_values("quality", ascending=False).reset_index(drop=True)
    s.to_csv(os.path.join(a.out_dir, "episode_scores.csv"), index=False)

    n_total = len(s)
    k = max(1, int(round(a.keep_frac * n_total)))
    rng = np.random.default_rng(a.seed)
    all_eps = sorted(s["episode_index"].tolist())
    subsets = {
        "full": all_eps,
        "curated": sorted(s["episode_index"].head(k).tolist()),
        "random": sorted(rng.choice(all_eps, size=k, replace=False).tolist()),
        "bottom": sorted(s["episode_index"].tail(k).tolist()),
    }
    subsets = {name: [int(e) for e in eps] for name, eps in subsets.items()}
    with open(os.path.join(a.out_dir, "subsets.json"), "w") as f:
        json.dump(subsets, f)

    print(f"Scored {n_total} episodes. Subset size for curated/random/bottom: {k}")
    print(s[["episode_index", "quality", "max_reward", "success", "length", "idle_frac", "jerk_ratio"]].head(10).to_string(index=False))
    print("...")
    print(s[["episode_index", "quality", "max_reward", "success", "length", "idle_frac", "jerk_ratio"]].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
