"""Step 1: compare the variety of the curated, random, and full subsets from trial 1."""
import argparse
import json

import numpy as np

from winnow_data import dataset_root, load_episodes


def coverage(points, lo, hi, bins):
    """Fraction of grid cells (over the dataset's range) that the points visit."""
    span = np.maximum(hi - lo, 1e-9)
    cells = np.clip(((points - lo) / span * bins).astype(int), 0, bins - 1)
    return len({tuple(c) for c in cells}) / bins ** points.shape[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", default="lerobot/pusht")
    p.add_argument("--root", default=None)
    p.add_argument("--subsets", default="trial_outputs/subsets.json")
    a = p.parse_args()

    eps = load_episodes(dataset_root(a.repo_id, a.root))
    subsets = json.load(open(a.subsets))
    all_act = np.concatenate([e["action"] for e in eps.values()])
    starts_all = np.stack([e["state"][0] for e in eps.values()])
    alo, ahi = all_act.min(0), all_act.max(0)
    slo, shi = starts_all.min(0), starts_all.max(0)

    print(f"{'subset':<10}{'episodes':>9}{'frames':>8}{'mean len':>10}{'len std':>9}"
          f"{'start spread':>14}{'start cover':>13}{'action cover':>14}")
    for name, ids in subsets.items():
        sel = [eps[i] for i in ids if i in eps]
        lengths = np.array([len(e["action"]) for e in sel])
        starts = np.stack([e["state"][0] for e in sel])
        acts = np.concatenate([e["action"] for e in sel])
        print(f"{name:<10}{len(sel):>9}{lengths.sum():>8}{lengths.mean():>10.1f}{lengths.std():>9.1f}"
              f"{np.linalg.norm(starts.std(0)):>14.1f}{coverage(starts, slo, shi, 6):>13.2f}"
              f"{coverage(acts, alo, ahi, 24):>14.2f}")
    print("\nstart spread: how far apart the starting positions are (higher = more varied)")
    print("start/action cover: fraction of the space the subset visits (higher = more varied)")


if __name__ == "__main__":
    main()
