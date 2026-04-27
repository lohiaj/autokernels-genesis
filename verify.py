#!/usr/bin/env python3
"""
verify.py -- end-to-end acceptance gate (FROZEN -- the agent NEVER modifies this).

Runs the team's production-grade regression at full scale. Slow (~30 min).

  1. Clear all kernel caches.
  2. Full pytest: tests/test_rigid_physics.py, the team-required subset, with the three
     known-flaky tests excluded (test_convexify, test_mesh_repair, test_mesh_primitive_COM).
  3. Full benchmark: benchmark_scaling.py --precision 32 --max-envs 8192 --num-steps 500.
  4. Compare e2e_throughput against the campaign's baseline_e2e_throughput in
     workspace/orchestration_state.json.
  5. Print a structured contract similar to bench.py.

Usage:
  uv run verify.py --campaign func_broad_phase

Exit codes:
  0  -- verify completed (PASS or FAIL printed in contract)
  1  -- harness error (container missing, etc.)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR / "workspace"
H100_REF = 794280.0
PYTEST_TIMEOUT = 1800   # 30 min
E2E_TIMEOUT = 1200      # 20 min


def container() -> str:
    name = os.environ.get("AUTOKERNEL_CONTAINER")
    if name:
        return name
    return f"ak-gpu{os.environ.get('AUTOKERNEL_GPU_ID', '0')}"


def docker_exec(cmd: str, timeout: int) -> tuple[int, str]:
    full = ["docker", "exec", container(), "bash", "-lc", cmd]
    try:
        r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--skip-pytest", action="store_true",
                    help="skip the regression suite (debugging only)")
    args = ap.parse_args()

    print("verify: wiping kernel caches...")
    docker_exec("rm -rf /root/.cache/quadrants /root/.cache/mesa_shader_cache 2>/dev/null; true", 60)

    if args.skip_pytest:
        print("verify: --skip-pytest set; treating regression as PASS")
        regression = "PASS"
        reg_tail = ""
    else:
        print("verify: running full regression (test_rigid_physics, ~10-15 min)...")
        cmd = (
            'cd /src/Genesis && GS_FAST_MATH=0 '
            f'timeout {PYTEST_TIMEOUT - 60} '
            'python -m pytest tests/test_rigid_physics.py -v -n 0 --forked -m required '
            '-k "not (test_convexify or test_mesh_repair or test_mesh_primitive_COM)" '
            '2>&1 | tail -200'
        )
        rc, out = docker_exec(cmd, PYTEST_TIMEOUT)
        regression = "PASS" if rc == 0 else "FAIL"
        reg_tail = out[-1500:]
        print(f"verify: regression={regression}")
        if regression == "FAIL":
            print(reg_tail)

    if regression == "PASS":
        print("verify: running full e2e benchmark (8192/500/FP32, ~5 min)...")
        json_out = "/tmp/verify_e2e.json"
        cmd = (
            f'cd /src/Genesis && GS_FAST_MATH=0 timeout {E2E_TIMEOUT - 30} '
            f'python benchmark_scaling.py --precision 32 --max-envs 8192 --num-steps 500 '
            f'--scaling-results-out {json_out} 2>&1 | tail -10'
        )
        t0 = time.time()
        rc, out = docker_exec(cmd, E2E_TIMEOUT)
        dt = time.time() - t0
        if rc != 0:
            print(f"verify: e2e CRASH (rc={rc}): {out[-800:]}")
            print()
            print("---")
            print(f"verify_status:      FAIL_E2E_CRASH")
            print(f"regression:         {regression}")
            return 0
        rc2, raw = docker_exec(f"cat {json_out}", 30)
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            print("verify: could not parse e2e JSON")
            print()
            print("---")
            print(f"verify_status:      FAIL_PARSE")
            print(f"regression:         {regression}")
            return 0
        entry = parsed if isinstance(parsed, dict) else (parsed[-1] if parsed else None)
        thr = entry.get("throughput") or entry.get("env_steps_per_sec") or 0.0
        wall = entry.get("wall_time_s") or entry.get("wall") or 0.0
        pct = 100.0 * thr / H100_REF
    else:
        thr = 0.0
        wall = 0.0
        pct = 0.0

    # Compare against orchestrator's baseline if known
    baseline = None
    state_path = WORKSPACE / "orchestration_state.json"
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        baseline = state.get("baseline_e2e_throughput")
    delta_pct = (100.0 * (thr - baseline) / baseline) if (baseline and thr) else 0.0
    overall = "PASS" if (regression == "PASS" and (baseline is None or thr >= baseline)) else "FAIL"

    print()
    print("---")
    print(f"verify_status:      {overall}")
    print(f"regression:         {regression}")
    print(f"e2e_throughput:     {thr:.0f}")
    print(f"e2e_wall_seconds:   {wall:.3f}")
    print(f"e2e_pct_of_h100:    {pct:.2f}")
    if baseline:
        print(f"baseline_e2e:       {baseline:.0f}")
        print(f"delta_vs_baseline:  {delta_pct:+.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
