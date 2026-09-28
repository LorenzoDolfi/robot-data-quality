"""
Dose-response setup: fixed-size training sets that differ only in how much of the data is corrupted.

Needs a corrupted dataset built with --frac 0.5, so there are enough clean and enough
corrupted episodes to fill a subset of --size either way.
"""
import argparse
import json
import os

import numpy as np

from winnow_data import dataset_root, load_episodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/pusht_dose")
    p.add_argument("--manifest", default=None)
    p.add_argument("--out_dir", default="trial3_outputs")
    p.add_argument("--size", type=int, default=100)
    p.add_argument("--fractions", default="0,0.25,0.5,1.0")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    manifest = a.manifest or os.path.join(a.root, "corruption_manifest.json")
    bad = {int(k) for k in json.load(open(manifest))}
    ids = sorted(load_episodes(dataset_root(None, a.root)))
    good = [e for e in ids if e not in bad]
    bad = sorted(bad)
    print(f"{len(ids)} episodes: {len(good)} clean, {len(bad)} corrupted")

    rng = np.random.default_rng(a.seed)
    # Draw from fixed shuffles so every subset reuses the same pool, changing only the mix.
    good_pool = list(rng.permutation(good))
    bad_pool = list(rng.permutation(bad))
    subsets = {}
    for f in [float(x) for x in a.fractions.split(",")]:
        n_bad = int(round(f * a.size))
        n_good = a.size - n_bad
        if n_bad > len(bad_pool) or n_good > len(good_pool):
            raise SystemExit(f"Not enough episodes for fraction {f}: need {n_good} clean and {n_bad} corrupted")
        name = f"c{int(round(f * 100)):03d}"
        subsets[name] = sorted(int(e) for e in good_pool[:n_good] + bad_pool[:n_bad])
    with open(os.path.join(a.out_dir, "subsets.json"), "w") as fh:
        json.dump(subsets, fh)
    for k, v in subsets.items():
        print(f"{k}: {len(v)} episodes, {len([e for e in v if e in set(bad)])} corrupted")
    print(f"Wrote {a.out_dir}/subsets.json")


if __name__ == "__main__":
    main()
