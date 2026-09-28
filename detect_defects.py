"""
Step 3: flag defective episodes using physical consistency, without looking at the manifest.

Idea: in PushT the action is the target position for the agent, so the agent should move
toward it. Real defects break that link between what the robot was told and what it did.
  consistency  correlation between the agent's velocity and (action - position)
  lag          time offset where that correlation peaks (clock drift shows up here)
  hf_noise     jitter in the actions relative to how much the agent moves
  freeze       longest stretch where the action is fixed but the agent keeps moving
Outputs flags.csv, a precision/recall report if a manifest is given, and training subsets.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from winnow_data import dataset_root, load_episodes

MAX_LAG = 10


def corr(x, y):
    x = x.ravel() - x.mean(); y = y.ravel() - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else 0.0


def features(a, s):
    v = s[1:] - s[:-1]
    n = len(v)
    lag_corr = {}
    for k in range(-MAX_LAG, MAX_LAG + 1):
        t = np.arange(max(0, -k), min(n, n - k))
        if len(t) > 5:
            lag_corr[k] = corr(v[t], a[t + k] - s[t])
    step_a = np.linalg.norm(np.diff(a, axis=0), axis=1)
    step_s = np.linalg.norm(v, axis=1)
    jitter = np.linalg.norm(np.diff(a, n=2, axis=0), axis=1)
    moving = step_s > np.median(step_s) * 0.5
    stuck = (step_a[: len(moving)] < 1e-6) & moving
    run = best = 0
    for x in stuck:
        run = run + 1 if x else 0
        best = max(best, run)
    return lag_corr, float(np.median(jitter) / (np.median(step_s) + 1e-9)), best / max(1, n)


def robust_z(x):
    x = np.asarray(x, float)
    mad = np.median(np.abs(x - np.median(x))) * 1.4826 + 1e-9
    return (x - np.median(x)) / mad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", default="lerobot/pusht")
    p.add_argument("--root", default=None)
    p.add_argument("--manifest", default=None)
    p.add_argument("--out_dir", default="trial2_outputs")
    p.add_argument("--z", type=float, default=3.5, help="robust z-score threshold")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    eps = load_episodes(dataset_root(a.repo_id, a.root))
    ids = sorted(eps)
    rows, lagcurves = [], []
    for e in ids:
        lc, hf, fz = features(eps[e]["action"], eps[e]["state"])
        lagcurves.append(lc)
        rows.append({"episode_index": e, "hf_noise": hf, "freeze_frac": fz})
    df = pd.DataFrame(rows)

    # The dataset's typical lag is the reference; each episode is judged at that lag and at its own best lag.
    ref_lag = int(pd.Series([max(lc, key=lc.get) for lc in lagcurves]).mode()[0])
    df["best_lag"] = [max(lc, key=lc.get) for lc in lagcurves]
    df["consistency"] = [lc.get(ref_lag, 0.0) for lc in lagcurves]
    df["lag_error"] = (df["best_lag"] - ref_lag).abs()

    z_cons = -robust_z(df["consistency"])
    z_hf = robust_z(df["hf_noise"])
    reasons = []
    for i in range(len(df)):
        r = []
        if z_cons[i] > a.z: r.append("actions don't match motion")
        if df["lag_error"][i] >= 3: r.append(f"time offset of {df['best_lag'][i] - ref_lag} steps")
        if z_hf[i] > a.z: r.append("noisy actions")
        if df["freeze_frac"][i] > 0.2: r.append("frozen actions")
        reasons.append("; ".join(r))
    df["reason"] = reasons
    df["flagged"] = df["reason"] != ""
    df.to_csv(os.path.join(a.out_dir, "flags.csv"), index=False)

    flagged = set(df.loc[df["flagged"], "episode_index"].astype(int))
    print(f"Reference lag: {ref_lag} steps. Flagged {len(flagged)} of {len(ids)} episodes.")

    rng = np.random.default_rng(a.seed)
    subsets = {"all": ids, "detector": [e for e in ids if e not in flagged]}
    if a.manifest:
        truth = {int(k): v for k, v in json.load(open(a.manifest)).items()}
        bad = set(truth)
        tp = len(flagged & bad)
        prec = tp / len(flagged) if flagged else 0.0
        rec = tp / len(bad) if bad else 0.0
        print(f"Precision {prec:.2f} (flags that were real defects), recall {rec:.2f} (defects that were caught)")
        for kind in sorted(set(truth.values())):
            ks = {e for e, k in truth.items() if k == kind}
            print(f"  {kind:<7} caught {len(ks & flagged)}/{len(ks)}")
        subsets["oracle"] = [e for e in ids if e not in bad]
        removed = rng.choice(ids, size=len(flagged), replace=False)
        subsets["random_removed"] = [e for e in ids if e not in set(removed.tolist())]
    subsets = {k: [int(x) for x in v] for k, v in subsets.items()}
    with open(os.path.join(a.out_dir, "subsets.json"), "w") as fh:
        json.dump(subsets, fh)
    print("Subset sizes:", {k: len(v) for k, v in subsets.items()})
    print(f"Wrote {a.out_dir}/flags.csv and {a.out_dir}/subsets.json")


if __name__ == "__main__":
    main()
