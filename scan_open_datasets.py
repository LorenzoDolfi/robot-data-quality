"""
Measure the real defect rate in public LeRobot datasets.

Downloads only the tabular data files (no videos), fits the simplest control model that
explains each dataset, then flags episodes whose actions disagree with the robot's motion.

  python scan_open_datasets.py --repo_id lerobot/pusht --repo_id lerobot/aloha_sim_insertion_human
  python scan_open_datasets.py --local_root data/pusht_corrupt        # test on known-bad data
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

MAX_LAG = 10


def corr(x, y):
    x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel()
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else 0.0


def robust_z(x):
    x = np.asarray(x, float)
    mad = np.median(np.abs(x - np.median(x))) * 1.4826 + 1e-9
    return (x - np.median(x)) / mad


def fetch(repo_id, max_files, cache_dir):
    """Download meta plus the first data files of a hub dataset; return local parquet paths."""
    from huggingface_hub import HfApi, hf_hub_download
    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    parquets = sorted(f for f in files if f.startswith("data/") and f.endswith(".parquet"))
    if not parquets:
        raise SystemExit(f"{repo_id}: no data parquet files found")
    picked = parquets[:max_files]
    print(f"{repo_id}: {len(parquets)} data files, downloading {len(picked)}")
    return [hf_hub_download(repo_id, f, repo_type="dataset", cache_dir=cache_dir) for f in picked]


def local_files(root):
    import glob
    return sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))


def episodes_from(paths):
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        keep = [c for c in ["episode_index", "frame_index", "action", "observation.state"] if c in df.columns]
        if "action" not in keep or "observation.state" not in keep:
            raise SystemExit(f"{p}: needs both 'action' and 'observation.state' columns")
        frames.append(df[keep])
    df = pd.concat(frames, ignore_index=True)
    eps = {}
    for ep, g in df.groupby("episode_index", sort=True):
        g = g.sort_values("frame_index") if "frame_index" in g else g
        a = np.stack(g["action"].to_numpy()).astype(float)
        s = np.stack(g["observation.state"].to_numpy()).astype(float)
        d = min(a.shape[1], s.shape[1])          # compare only the dimensions they share
        if len(a) > 20:
            eps[int(ep)] = (a[:, :d], s[:, :d])
    return eps


def lag_curve(a, s, model):
    """Correlation between the robot's motion and the command, at each time offset."""
    v = s[1:] - s[:-1]
    n = len(v)
    out = {}
    for k in range(-MAX_LAG, MAX_LAG + 1):
        t = np.arange(max(0, -k), min(n, n - k))
        if len(t) > 5:
            cmd = a[t + k] - s[t] if model == "position" else a[t + k]
            out[k] = corr(v[t], cmd)
    return out


def scan(name, eps, z_thresh, min_consistency):
    # Pick the control model that explains this dataset better: actions as target positions,
    # or actions as velocity commands.
    sample = list(eps)[: min(40, len(eps))]
    scores = {m: np.median([max(lag_curve(*eps[e], m).values()) for e in sample]) for m in ("position", "velocity")}
    model = max(scores, key=scores.get)
    curves = {e: lag_curve(*eps[e], model) for e in eps}
    ref_lag = int(pd.Series([max(c, key=c.get) for c in curves.values()]).mode()[0])

    rows = []
    for e, (a, s) in eps.items():
        step_s = np.linalg.norm(s[1:] - s[:-1], axis=1)
        step_a = np.linalg.norm(np.diff(a, axis=0), axis=1)
        jitter = np.linalg.norm(np.diff(a, n=2, axis=0), axis=1)
        moving = step_s > np.median(step_s) * 0.5
        stuck = (step_a[: len(moving)] < 1e-6) & moving
        run = best = 0
        for x in stuck:
            run = run + 1 if x else 0
            best = max(best, run)
        rows.append({"episode_index": e, "consistency": curves[e].get(ref_lag, 0.0),
                     "best_lag": max(curves[e], key=curves[e].get),
                     "hf_noise": float(np.median(jitter) / (np.median(step_s) + 1e-9)),
                     "freeze_frac": best / max(1, len(step_s))})
    df = pd.DataFrame(rows)
    df["lag_error"] = (df["best_lag"] - ref_lag).abs()

    zc, zh = -robust_z(df["consistency"]), robust_z(df["hf_noise"])
    reasons = []
    for i in range(len(df)):
        r = []
        # Relative outlier AND absolutely poor tracking, so uniformly noisy datasets aren't all flagged.
        if zc[i] > z_thresh and df["consistency"][i] < min_consistency: r.append("actions don't match motion")
        if df["lag_error"][i] >= 3: r.append(f"time offset {int(df['best_lag'][i] - ref_lag)} steps")
        if zh[i] > z_thresh and df["hf_noise"][i] > 1.0: r.append("noisy actions")
        if df["freeze_frac"][i] > 0.2: r.append("frozen actions")
        reasons.append("; ".join(r))
    df["reason"] = reasons
    df["flagged"] = df["reason"] != ""

    print(f"\n=== {name} ===")
    print(f"  control model: actions look like {model} commands "
          f"(fit {scores[model]:.2f} vs {scores['velocity' if model == 'position' else 'position']:.2f})")
    print(f"  episodes scanned: {len(df)}   reference lag: {ref_lag}")
    print(f"  median consistency: {df['consistency'].median():.3f}   "
          f"10th percentile: {df['consistency'].quantile(0.1):.3f}")
    print(f"  FLAGGED: {int(df['flagged'].sum())} ({100 * df['flagged'].mean():.1f}%)")
    for r in sorted({x for row in df['reason'] for x in row.split('; ') if x}):
        print(f"    {r}: {sum(r in row for row in df['reason'])}")
    if scores[model] < 0.3:
        print("  NOTE: neither control model fits well, so these flags are unreliable for this dataset.")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", action="append", default=[])
    p.add_argument("--local_root", default=None)
    p.add_argument("--max_files", type=int, default=3, help="data files per dataset (keeps downloads small)")
    p.add_argument("--out_dir", default="scan_results")
    p.add_argument("--z", type=float, default=3.5)
    p.add_argument("--min_consistency", type=float, default=0.5)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    jobs = [(a.local_root, local_files(a.local_root))] if a.local_root else \
           [(r, fetch(r, a.max_files, os.path.join(a.out_dir, "cache"))) for r in a.repo_id]
    if not jobs:
        raise SystemExit("Pass --repo_id (one or more) or --local_root")

    summary = {}
    for name, paths in jobs:
        eps = episodes_from(paths)
        if not eps:
            print(f"{name}: no usable episodes"); continue
        df = scan(name, eps, a.z, a.min_consistency)
        slug = name.replace("/", "_").strip("._")
        df.to_csv(os.path.join(a.out_dir, f"{slug}.csv"), index=False)
        summary[name] = {"episodes": len(df), "flagged": int(df["flagged"].sum()),
                         "rate": round(float(df["flagged"].mean()), 4)}
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("\nDefect rate by dataset:")
    for k, v in summary.items():
        print(f"  {k}: {100 * v['rate']:.1f}%  ({v['flagged']}/{v['episodes']})")
    print(f"\nPer-episode details in {a.out_dir}/")


if __name__ == "__main__":
    main()
