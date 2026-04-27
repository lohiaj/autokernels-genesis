#!/usr/bin/env python3
"""
analysis.py -- morning dashboard for autokernels-genesis.

Reads results.tsv (in the current directory or specified by --results), generates progress.png
showing the running min of kernel_avg_us and running max of e2e_throughput.

Usage:
  uv run analysis.py
  uv run analysis.py --results /path/to/results.tsv --out progress.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.tsv")
    ap.add_argument("--out", default="progress.png")
    args = ap.parse_args()

    p = Path(args.results)
    if not p.exists():
        print(f"analysis: ERROR: {p} not found", file=sys.stderr)
        return 1

    df = pd.read_csv(p, sep="\t")
    df["experiment"] = pd.to_numeric(df["experiment"], errors="coerce")
    df["kernel_avg_us"] = pd.to_numeric(df["kernel_avg_us"], errors="coerce")
    df["e2e_throughput"] = pd.to_numeric(df["e2e_throughput"], errors="coerce")
    df["correctness"] = df["correctness"].astype(str).str.upper()

    valid = df[df["correctness"] == "PASS"].copy()
    if valid.empty:
        print("analysis: no PASS rows")
        return 0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Kernel avg us (lower is better) -- running min
    ax1.scatter(valid["experiment"], valid["kernel_avg_us"], c="lightgray", s=20, label="all PASS")
    kept = valid[valid["description"].str.lower().str.startswith("baseline") |
                 (valid["kernel_avg_us"].cummin() == valid["kernel_avg_us"])]
    ax1.scatter(kept["experiment"], kept["kernel_avg_us"], c="green", s=50,
                edgecolors="black", linewidths=0.5, label="kept")
    running_min = valid["kernel_avg_us"].cummin()
    ax1.step(valid["experiment"], running_min, where="post", color="darkgreen",
             linewidth=2, alpha=0.6, label="running best")
    ax1.set_ylabel("kernel_avg_us (lower is better)")
    ax1.set_title("Targeted kernel per-call avg")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.2)

    # E2E throughput (higher is better) -- running max
    ax2.scatter(valid["experiment"], valid["e2e_throughput"], c="lightgray", s=20)
    running_max = valid["e2e_throughput"].cummax()
    ax2.step(valid["experiment"], running_max, where="post", color="darkblue",
             linewidth=2, alpha=0.6, label="running best")
    ax2.axhline(794280, color="red", linestyle="--", alpha=0.5, label="H100 reference")
    ax2.axhline(794280 * 0.8, color="orange", linestyle="--", alpha=0.5, label="80% H100 target")
    ax2.set_xlabel("experiment #")
    ax2.set_ylabel("e2e_throughput (env*steps/s, higher is better)")
    ax2.set_title("End-to-end throughput")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"saved {args.out}")

    # Terminal summary
    print()
    print(f"experiments:        {len(df)}")
    print(f"PASS:               {len(valid)}")
    print(f"best kernel_avg_us: {valid['kernel_avg_us'].min():.2f}")
    print(f"best e2e:           {valid['e2e_throughput'].max():.0f} env*steps/s")
    print(f"% of H100:          {100 * valid['e2e_throughput'].max() / 794280:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
