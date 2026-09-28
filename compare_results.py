"""Collect eval results into one table, and average across seeds when there are several."""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def find_metrics(obj):
    if isinstance(obj, dict):
        if "pc_success" in obj or "avg_max_reward" in obj:
            return obj
        for key in ("aggregated", "overall"):
            if key in obj and (m := find_metrics(obj[key])):
                return m
        for v in obj.values():
            if m := find_metrics(v):
                return m
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="trial_outputs")
    a = p.parse_args()
    sp = os.path.join(a.out_dir, "subsets.json")
    subsets = json.load(open(sp)) if os.path.exists(sp) else {}

    rows = []
    for d in sorted(glob.glob(os.path.join(a.out_dir, "eval_*"))):
        files = glob.glob(os.path.join(d, "**", "eval_info.json"), recursive=True)
        if not files:
            continue
        m = find_metrics(json.load(open(files[0]))) or {}
        tag = os.path.basename(d)[len("eval_"):]
        subset = max((k for k in subsets if tag.startswith(k + "_")), key=len, default=tag.rsplit("_", 1)[0])
        steps = tag[len(subset) + 1:].split("_")[0]
        rows.append((tag, subset, steps, len(subsets.get(subset, [])), m.get("pc_success"), m.get("avg_max_reward")))

    if not rows:
        print("No eval results found yet.")
        return
    fmt = lambda x, f: format(x, f) if isinstance(x, (int, float)) else "n/a"
    print(f"{'run':<30}{'episodes':>10}{'success %':>12}{'avg max reward':>18}")
    for tag, _, _, n, s, r in rows:
        print(f"{tag:<30}{n:>10}{fmt(s, '.1f'):>12}{fmt(r, '.3f'):>18}")

    groups = defaultdict(list)
    for _, subset, steps, _, s, r in rows:
        groups[(subset, steps)].append((s, r))
    if any(len(v) > 1 for v in groups.values()):
        print(f"\n{'subset (mean over seeds)':<30}{'seeds':>6}{'success %':>16}{'avg max reward':>20}")
        for (subset, steps), v in sorted(groups.items()):
            s = np.array([x[0] for x in v if x[0] is not None], float)
            r = np.array([x[1] for x in v if x[1] is not None], float)
            print(f"{subset + '_' + steps:<30}{len(v):>6}{s.mean():>10.1f} ± {s.std():<4.1f}{r.mean():>13.3f} ± {r.std():.3f}")


if __name__ == "__main__":
    main()
