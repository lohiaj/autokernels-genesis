#!/usr/bin/env python3
"""
correctness.py -- thin standalone wrapper around the scoped pytest subset.

bench.py already invokes this internally, but exposing it as a CLI lets the agent
re-run just the correctness suite after a small fix without paying for the full
benchmark cycle.

Usage:
  uv run correctness.py --campaign func_broad_phase
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
KERNELS = SCRIPT_DIR / "kernels"
TIMEOUT_S = 600


def container() -> str:
    name = os.environ.get("AUTOKERNEL_CONTAINER")
    if name:
        return name
    return f"ak-gpu{os.environ.get('AUTOKERNEL_GPU_ID', '0')}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()

    manifest_path = KERNELS / args.campaign / "target.json"
    if not manifest_path.exists():
        print(f"correctness: ERROR: {manifest_path} not found", file=sys.stderr)
        return 1
    with open(manifest_path) as f:
        m = json.load(f)
    tests = m.get("correctness_tests", [])
    if not tests:
        print("correctness: SKIP (no tests listed)")
        return 0

    test_args = " ".join(shlex.quote(t) for t in tests)
    cmd = (
        f"cd /src/Genesis && GS_FAST_MATH=0 timeout {TIMEOUT_S - 30} "
        f"python -m pytest {test_args} -v -n 0 --forked -m required 2>&1 | tail -200"
    )
    full = ["docker", "exec", container(), "bash", "-lc", cmd]
    try:
        r = subprocess.run(full, capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print("correctness: TIMEOUT")
        return 124
    print(r.stdout, end="")
    print(r.stderr, end="")
    print()
    print("---")
    if r.returncode == 0:
        print("correctness: PASS")
    else:
        print(f"correctness: FAIL (rc={r.returncode})")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
