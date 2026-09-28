"""
Step 2: copy a LeRobot dataset and corrupt a fraction of its episodes' action labels.

Corruptions only change the action column, so frame counts, videos, and metadata stay valid:
  noise   sensor or teleop noise added to every action
  freeze  actions stuck at one value for a long stretch (teleop dropout)
  shift   actions misaligned in time with the observations (clock drift)
  swap    actions taken from a different episode (mismatched files)
A manifest records which episodes were corrupted and how.
"""
import argparse
import json
import os
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from winnow_data import data_files, dataset_root

TYPES = ["noise", "freeze", "shift", "swap"]


def corrupt(kind, a, lo, hi, rng, donor=None):
    a = a.copy()
    n = len(a)
    if kind == "noise":
        a += rng.normal(0, 0.08 * (hi - lo), a.shape)
    elif kind == "freeze":
        s = int(rng.uniform(0.2, 0.4) * n)
        e = min(n, s + int(0.4 * n))
        a[s:e] = a[s]
    elif kind == "shift":
        k = min(6, n - 1)
        a = np.concatenate([a[k:], np.repeat(a[-1:], k, axis=0)])
    elif kind == "swap":
        src = np.linspace(0, len(donor) - 1, n)
        a = np.stack([np.interp(src, np.arange(len(donor)), donor[:, d]) for d in range(a.shape[1])], 1)
    return np.clip(a, lo, hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_repo", default="lerobot/pusht")
    p.add_argument("--src_root", default=None)
    p.add_argument("--out_root", default="data/pusht_corrupt")
    p.add_argument("--frac", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    src = dataset_root(a.src_repo, a.src_root)
    out = os.path.abspath(a.out_root)
    if os.path.exists(out):
        if not a.overwrite:
            raise SystemExit(f"{out} already exists. Use --overwrite to replace it.")
        shutil.rmtree(out)
    shutil.copytree(src, out, symlinks=False)

    files = data_files(out)
    tables = {f: pq.read_table(f) for f in files}
    ep_rows, all_actions = {}, []
    for f, t in tables.items():
        ep = t.column("episode_index").to_numpy()
        fr = t.column("frame_index").to_numpy()
        act = np.array(t.column("action").to_pylist(), dtype=np.float64)
        all_actions.append(act)
        for e in np.unique(ep):
            idx = np.where(ep == e)[0]
            ep_rows[int(e)] = (f, idx[np.argsort(fr[idx])])
    all_act = np.concatenate(all_actions)
    lo, hi = all_act.min(0), all_act.max(0)

    rng = np.random.default_rng(a.seed)
    ids = sorted(ep_rows)
    n_bad = int(round(a.frac * len(ids)))
    bad = sorted(rng.choice(ids, size=n_bad, replace=False).tolist())
    kinds = [TYPES[i % len(TYPES)] for i in range(n_bad)]
    rng.shuffle(kinds)
    manifest = {int(e): k for e, k in zip(bad, kinds)}
    clean = [e for e in ids if e not in manifest]

    actions = {f: np.array(t.column("action").to_pylist(), dtype=np.float64) for f, t in tables.items()}
    originals = {e: actions[f][rows].copy() for e, (f, rows) in ep_rows.items()}
    for e, kind in manifest.items():
        f, rows = ep_rows[e]
        donor = originals[int(rng.choice(clean))] if kind == "swap" else None
        actions[f][rows] = corrupt(kind, originals[e], lo, hi, rng, donor)

    for f, t in tables.items():
        i = t.schema.get_field_index("action")
        field = t.schema.field(i)
        col = pa.array(actions[f].astype(np.float32).tolist(), type=field.type)
        pq.write_table(t.set_column(i, field, col), f)

    with open(os.path.join(out, "corruption_manifest.json"), "w") as fh:
        json.dump({str(k): v for k, v in manifest.items()}, fh, indent=1)
    counts = {k: kinds.count(k) for k in TYPES}
    print(f"Copied {len(ids)} episodes to {out}")
    print(f"Corrupted {n_bad} episodes: {counts}")
    print(f"Manifest: {os.path.join(out, 'corruption_manifest.json')}")


if __name__ == "__main__":
    main()
