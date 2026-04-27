#!/usr/bin/env python3
"""
bench.py -- autokernels-genesis benchmark harness (FROZEN -- the agent NEVER modifies this).

What this does, in order:
  1. Resolve the campaign manifest (kernels/<campaign>/target.json) and verify edit_files exist.
  2. Inside the pinned container: wipe Quadrants/Mesa kernel caches.
  3. Run the scoped pytest correctness subset for this campaign. If FAIL -> abort, print contract.
  4. Run an UNTRACED bench (8192 envs, 500 steps, FP32) for headline e2e throughput.
  5. Run a TRACED bench (8192 envs, 100 steps, FP32) under rocprofv3 for per-kernel attribution.
  6. Parse rocprofv3 _kernel_stats.csv, sum entries matching target.json::primary_kernel_pattern.
     Drop runtime_initialize_rand_states_serial (one-shot init, not per-step).
  7. Print the structured contract for the agent to grep.

Usage (run on the host, talks to a docker container via `docker exec`):
  uv run bench.py --campaign func_broad_phase
  uv run bench.py --campaign kernel_step_1_2 --skip-traced     # skip rocprofv3 (faster, no per-kernel attribution)
  uv run bench.py --campaign func_broad_phase --profile-omniperf   # heavier omniperf profile

Environment variables (set by launcher/launch_8gpu.sh):
  AUTOKERNEL_GPU_ID       -- 0..7, the GPU pinned to this container
  AUTOKERNEL_CONTAINER    -- docker container name (e.g. ak-gpu0)
  GENESIS_SRC             -- host path to the Genesis worktree this agent edits
  AUTOKERNEL_WT           -- host path to this agent's autokernels-genesis worktree
  AUTOKERNEL_ROOT              -- host path to ~/work (for newton-assets, quadrants)

Exit codes:
  0  -- bench completed (correctness may still be FAIL; see contract output)
  1  -- harness error (container missing, target.json malformed, etc.)
  2  -- bench timed out

H100 reference (constant): 794280 env*steps/s @ 8192/500/FP32 -- see project_context.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

H100_THROUGHPUT_REF = 794280.0   # env*steps/s, from project_context.md
DEFAULT_E2E_NSTEPS = 500         # untraced (headline)
DEFAULT_PROF_NSTEPS = 100        # traced (per-kernel attribution; rocprofv3 has ~30% overhead)
DEFAULT_NENVS = 8192
DEFAULT_PRECISION = "32"

UNTRACED_TIMEOUT_S = 600         # 10 min hard ceiling for untraced bench
TRACED_TIMEOUT_S = 900           # 15 min for traced (slower, more steps in profile mode)
CORRECTNESS_TIMEOUT_S = 600      # 10 min for scoped pytest

EXCLUDE_KERNEL_RE = re.compile(r"runtime_initialize_rand_states")

# Campaign manifest path relative to repo root
SCRIPT_DIR = Path(__file__).resolve().parent
KERNELS_DIR = SCRIPT_DIR / "kernels"
WORKSPACE_DIR = SCRIPT_DIR / "workspace"
RUNS_DIR = SCRIPT_DIR / "runs"

# ---------------------------------------------------------------------------
# Container exec wrapper
# ---------------------------------------------------------------------------

def _container() -> str:
    name = os.environ.get("AUTOKERNEL_CONTAINER")
    if not name:
        # Default per-GPU naming if not set explicitly
        gpu = os.environ.get("AUTOKERNEL_GPU_ID", "0")
        name = f"ak-gpu{gpu}"
    return name

def _genesis_src() -> str:
    src = os.environ.get("GENESIS_SRC")
    if not src:
        src = os.path.expanduser("~/work/Genesis")
    return src

def _docker_exec(cmd: str, timeout: int) -> tuple[int, str]:
    """Run `cmd` inside the container. Returns (returncode, combined_stdout_stderr)."""
    full = ["docker", "exec", _container(), "bash", "-lc", cmd]
    try:
        r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as e:
        return 124, f"TIMEOUT after {timeout}s: {e}"
    except FileNotFoundError:
        return 127, "docker not found on host PATH"

def _container_exists(name: str) -> bool:
    r = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name],
        capture_output=True, text=True,
    )
    return r.returncode == 0 and r.stdout.strip() == "true"

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def load_manifest(campaign: str) -> dict:
    path = KERNELS_DIR / campaign / "target.json"
    if not path.exists():
        die(f"campaign manifest not found: {path}")
    with open(path) as f:
        m = json.load(f)
    required = ["campaign_id", "primary_kernel_pattern", "edit_files",
                "current_pct_of_h100", "correctness_tests"]
    for k in required:
        if k not in m:
            die(f"manifest missing required key: {k}")
    return m

def verify_edit_files_exist(manifest: dict) -> None:
    src = _genesis_src()
    for ef in manifest["edit_files"]:
        full = os.path.join(src, ef["path"])
        if not os.path.exists(full):
            die(f"edit_file does not exist on host: {full} -- "
                f"is GENESIS_SRC={src} pointing at a Genesis checkout?")

# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def wipe_caches() -> None:
    """Wipe Quadrants and Mesa kernel caches inside the container.

    Per project_context.md: every Genesis edit requires this or the cached compiled kernel
    is reused and the patch silently no-ops.
    """
    rc, out = _docker_exec(
        "rm -rf /root/.cache/quadrants /root/.cache/mesa_shader_cache 2>/dev/null; true",
        timeout=60,
    )
    if rc != 0:
        warn(f"cache wipe returned rc={rc}: {out[-300:]}")


def run_correctness(manifest: dict) -> tuple[str, str]:
    """Run the scoped pytest subset. Returns (status, log_tail)."""
    tests = manifest["correctness_tests"]
    if not tests:
        return "SKIP", "no correctness_tests listed"
    test_args = " ".join(shlex.quote(t) for t in tests)
    # Match team's flags: -v -n 0 --forked -m required (per dev guidelines)
    # Keep the output captured but small.
    cmd = (
        f"cd /src/Genesis && "
        f"GS_FAST_MATH=0 timeout {CORRECTNESS_TIMEOUT_S - 30} "
        f"python -m pytest {test_args} -v -n 0 --forked -m required 2>&1 | tail -200"
    )
    rc, out = _docker_exec(cmd, timeout=CORRECTNESS_TIMEOUT_S)
    if rc == 124:
        return "TIMEOUT", out[-400:]
    if rc == 0:
        return "PASS", out[-400:]
    if "FAILED" in out or "ERROR" in out:
        return "FAIL", out[-1000:]
    return "FAIL", f"rc={rc}\n{out[-800:]}"


def run_untraced_bench(out_dir: str) -> dict:
    """Run /work/bench_mi300.py untraced for headline e2e numbers.

    bench_mi300.py is the team's purpose-built single-config harness (vs benchmark_scaling.py
    which sweeps multiple n_envs and writes to a fixed path). Faster for our iteration loop.
    """
    json_out = f"{out_dir}/untraced.json"
    cmd = (
        f"mkdir -p {out_dir} && "
        f"GS_FAST_MATH=0 "
        f"PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless PYGLET_HEADLESS=true "
        f"timeout {UNTRACED_TIMEOUT_S - 30} "
        f"python3 /work/bench_mi300.py "
        f"--precision {DEFAULT_PRECISION} "
        f"--n-envs {DEFAULT_NENVS} "
        f"--num-steps {DEFAULT_E2E_NSTEPS} "
        f"--tag untraced "
        f"--out {json_out} 2>&1 | tail -30"
    )
    t0 = time.time()
    rc, out = _docker_exec(cmd, timeout=UNTRACED_TIMEOUT_S)
    dt = time.time() - t0

    if rc == 124:
        return {"status": "TIMEOUT", "wall_seconds": dt, "log_tail": out[-800:]}
    if rc != 0:
        return {"status": "CRASH", "wall_seconds": dt, "log_tail": out[-1500:]}

    # Try to parse the JSON. benchmark_scaling.py writes a list of entries; the 8192-env
    # entry is the one we want.
    parsed = _read_json_in_container(json_out)
    if parsed is None:
        return {"status": "PARSE_FAIL", "wall_seconds": dt, "log_tail": out[-800:]}

    # bench_mi300.py writes a single dict (not a list), with keys: throughput, wall_time_s, n_envs, ...
    entry = parsed if isinstance(parsed, dict) else _select_8192_entry(parsed)
    if entry is None:
        return {"status": "PARSE_FAIL", "wall_seconds": dt, "log_tail": out[-800:],
                "raw": str(parsed)[:400]}

    throughput = entry.get("throughput") or entry.get("env_steps_per_sec") or 0.0
    wall = entry.get("wall_time_s") or entry.get("wall") or 0.0
    return {
        "status": "PASS",
        "e2e_throughput": float(throughput),
        "e2e_wall_seconds": float(wall),
        "wall_seconds": dt,
        "log_tail": out[-200:],
    }


def run_traced_bench(out_dir: str) -> dict:
    """Run benchmark under rocprofv3 for per-kernel attribution."""
    cmd = (
        f"mkdir -p {out_dir} && cd {out_dir} && "
        f"GS_FAST_MATH=0 "
        f"PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless PYGLET_HEADLESS=true "
        f"timeout {TRACED_TIMEOUT_S - 60} "
        f"rocprofv3 --stats --kernel-trace -f csv -d . -o traced -- "
        f"python3 /work/bench_mi300.py "
        f"--precision {DEFAULT_PRECISION} "
        f"--n-envs {DEFAULT_NENVS} "
        f"--num-steps {DEFAULT_PROF_NSTEPS} "
        f"--tag traced "
        f"--out {out_dir}/traced.json 2>&1 | tail -30"
    )
    t0 = time.time()
    rc, out = _docker_exec(cmd, timeout=TRACED_TIMEOUT_S)
    dt = time.time() - t0

    if rc == 124:
        return {"status": "TIMEOUT", "wall_seconds": dt, "log_tail": out[-800:]}
    if rc != 0:
        return {"status": "CRASH", "wall_seconds": dt, "log_tail": out[-1500:]}

    return {"status": "PASS", "wall_seconds": dt, "csv_path": f"{out_dir}/traced_kernel_stats.csv"}


def parse_kernel_stats(csv_path: str, pattern: str) -> dict:
    """Parse rocprofv3's _kernel_stats.csv (inside container). Sum entries matching `pattern`,
    excluding `runtime_initialize_rand_states_serial` (one-shot init).

    Returns: {primary_total_ns, primary_calls, primary_avg_ns, peak_vram_mb (None if not in CSV),
              all_kernel_total_ns, top_5_kernels: [(name, total_ms, pct_of_all), ...]}
    """
    rc, content = _docker_exec(f"cat {csv_path} 2>&1", timeout=30)
    if rc != 0:
        return {"error": f"could not read {csv_path}: {content[:300]}"}

    rows = list(csv.DictReader(content.splitlines()))
    if not rows:
        return {"error": "kernel_stats CSV empty"}

    # Schema varies slightly across rocprofv3 versions; try common headers.
    name_keys = ["KernelName", "Kernel_Name", "Kernel name", "kernel_name", "Name"]
    total_keys = ["TotalDurationNs", "Total_Duration_Ns", "TotalNs", "Total"]
    calls_keys = ["Calls", "Count", "Invocations"]

    def _pick(row: dict, keys: list[str]) -> str:
        for k in keys:
            if k in row:
                return row[k]
        return ""

    pat_re = re.compile(pattern)
    primary_total_ns = 0.0
    primary_calls = 0
    all_total_ns = 0.0
    breakdown: list[tuple[str, float, int]] = []

    for row in rows:
        name = _pick(row, name_keys)
        if not name:
            continue
        if EXCLUDE_KERNEL_RE.search(name):
            continue
        try:
            total_ns = float(_pick(row, total_keys) or 0)
            calls = int(float(_pick(row, calls_keys) or 0))
        except ValueError:
            continue
        all_total_ns += total_ns
        breakdown.append((name, total_ns, calls))
        if pat_re.search(name):
            primary_total_ns += total_ns
            primary_calls += calls

    breakdown.sort(key=lambda r: r[1], reverse=True)
    top_5 = [
        (name, total_ns / 1e6, (100.0 * total_ns / all_total_ns) if all_total_ns else 0.0)
        for name, total_ns, _ in breakdown[:5]
    ]

    primary_avg_ns = (primary_total_ns / primary_calls) if primary_calls else 0.0
    return {
        "primary_total_ns": primary_total_ns,
        "primary_calls": primary_calls,
        "primary_avg_ns": primary_avg_ns,
        "all_kernel_total_ns": all_total_ns,
        "top_5_kernels": top_5,
    }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json_in_container(path: str) -> object | None:
    rc, content = _docker_exec(f"cat {path} 2>&1", timeout=30)
    if rc != 0:
        return None
    try:
        return json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return None


def _select_8192_entry(parsed: object) -> dict | None:
    """benchmark_scaling.py writes a list of size results; pick the n_envs=8192 entry."""
    if isinstance(parsed, dict):
        # single-config run
        return parsed
    if not isinstance(parsed, list):
        return None
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        n = entry.get("n_envs") or entry.get("num_envs") or entry.get("envs")
        if n == DEFAULT_NENVS or n == str(DEFAULT_NENVS):
            return entry
    # fall back to last entry if nothing matched explicitly
    return parsed[-1] if parsed else None


def _peak_vram_mb_in_container() -> float | None:
    """Best-effort: parse `rocm-smi --showmeminfo vram` for the pinned GPU."""
    rc, out = _docker_exec("rocm-smi --showmeminfo vram 2>&1", timeout=10)
    if rc != 0:
        return None
    # Output lines like: "GPU[0] : Used Memory (B): 142800000000"
    m = re.search(r"Used Memory \(B\):\s*(\d+)", out)
    if not m:
        return None
    return int(m.group(1)) / (1024 * 1024)


def die(msg: str) -> None:
    print(f"bench: ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def warn(msg: str) -> None:
    print(f"bench: WARN: {msg}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

def print_contract(*, correctness: str, kernel_avg_us: float, kernel_total_ms: float,
                   kernel_calls: int, e2e_throughput: float, e2e_wall_seconds: float,
                   peak_vram_mb: float, profile_overhead_pct: float,
                   top_5: list[tuple[str, float, float]]) -> None:
    print()
    print("---")
    print(f"correctness:        {correctness}")
    print(f"kernel_avg_us:      {kernel_avg_us:.2f}")
    print(f"kernel_total_ms:    {kernel_total_ms:.2f}")
    print(f"kernel_calls:       {kernel_calls}")
    print(f"e2e_throughput:     {e2e_throughput:.0f}")
    pct = (100.0 * e2e_throughput / H100_THROUGHPUT_REF) if H100_THROUGHPUT_REF else 0.0
    print(f"e2e_pct_of_h100:    {pct:.2f}")
    print(f"e2e_wall_seconds:   {e2e_wall_seconds:.3f}")
    print(f"peak_vram_mb:       {peak_vram_mb:.1f}")
    vpct = (100.0 * peak_vram_mb / (192 * 1024)) if peak_vram_mb else 0.0
    print(f"peak_vram_pct:      {vpct:.1f}")
    print(f"profile_overhead_pct: {profile_overhead_pct:.1f}")
    if top_5:
        print()
        print("top_5_kernels (post init-strip):")
        for name, ms, p in top_5:
            short = name if len(name) < 70 else name[:67] + "..."
            print(f"  {p:5.1f}%  {ms:8.2f} ms  {short}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True, help="kernels/<campaign>/ subdirectory name")
    ap.add_argument("--skip-traced", action="store_true",
                    help="skip rocprofv3 traced run (faster, no per-kernel attribution)")
    ap.add_argument("--skip-correctness", action="store_true",
                    help="WARNING: skip pytest. For dev/debug only.")
    ap.add_argument("--profile-omniperf", action="store_true",
                    help="run omniperf profile in addition (slow, for stuck campaigns)")
    args = ap.parse_args()

    # Sanity
    container = _container()
    if not _container_exists(container):
        die(f"container {container} is not running. Did you run launcher/launch_8gpu.sh?")
    if not os.path.isdir(_genesis_src()):
        die(f"GENESIS_SRC={_genesis_src()} is not a directory")

    manifest = load_manifest(args.campaign)
    verify_edit_files_exist(manifest)

    print(f"bench: campaign={args.campaign} container={container} gpu={os.environ.get('AUTOKERNEL_GPU_ID', '?')}")
    print(f"bench: editing genesis at {_genesis_src()}")
    print(f"bench: primary_kernel_pattern={manifest['primary_kernel_pattern']}")

    # Step 1: wipe caches
    print("bench: wiping kernel caches...")
    wipe_caches()

    # Step 2: correctness
    if args.skip_correctness:
        warn("--skip-correctness set; not running pytest")
        correctness = "SKIP"
    else:
        print("bench: running correctness suite...")
        correctness, ctail = run_correctness(manifest)
        print(f"bench: correctness={correctness}")
        if correctness != "PASS":
            print(f"bench: correctness output (tail):\n{ctail}")
            print_contract(
                correctness=correctness, kernel_avg_us=0, kernel_total_ms=0,
                kernel_calls=0, e2e_throughput=0, e2e_wall_seconds=0,
                peak_vram_mb=0, profile_overhead_pct=0, top_5=[],
            )
            return 0

    # Make a per-experiment workspace dir inside the container (mounted at /work/runs/<id>)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir_in_container = f"/work/runs/{args.campaign}-{run_id}"

    # Step 3: untraced bench (headline e2e)
    print("bench: untraced run (8192/500/FP32)...")
    untraced = run_untraced_bench(out_dir_in_container)
    if untraced["status"] != "PASS":
        print(f"bench: untraced {untraced['status']} -- {untraced.get('log_tail', '')}")
        print_contract(
            correctness="CRASH" if untraced["status"] == "CRASH" else "TIMEOUT",
            kernel_avg_us=0, kernel_total_ms=0, kernel_calls=0,
            e2e_throughput=0, e2e_wall_seconds=0,
            peak_vram_mb=0, profile_overhead_pct=0, top_5=[],
        )
        return 0
    e2e_thr = untraced["e2e_throughput"]
    e2e_wall = untraced["e2e_wall_seconds"]
    untraced_walltime = untraced["wall_seconds"]
    print(f"bench: untraced e2e_throughput={e2e_thr:.0f} env*steps/s, wall={e2e_wall:.2f}s")

    # Step 4: traced bench (per-kernel attribution)
    if args.skip_traced:
        warn("--skip-traced set; no per-kernel attribution")
        kernel_avg_us = 0.0
        kernel_total_ms = 0.0
        kernel_calls = 0
        top_5 = []
        prof_overhead = 0.0
    else:
        print("bench: traced run (8192/100/FP32 under rocprofv3)...")
        traced = run_traced_bench(out_dir_in_container)
        if traced["status"] != "PASS":
            warn(f"traced run {traced['status']}; treating as no-attribution")
            kernel_avg_us = 0.0
            kernel_total_ms = 0.0
            kernel_calls = 0
            top_5 = []
            prof_overhead = 0.0
        else:
            stats = parse_kernel_stats(traced["csv_path"], manifest["primary_kernel_pattern"])
            if "error" in stats:
                warn(f"parse_kernel_stats: {stats['error']}")
                kernel_avg_us = 0.0
                kernel_total_ms = 0.0
                kernel_calls = 0
                top_5 = []
            else:
                kernel_avg_us = stats["primary_avg_ns"] / 1000.0
                kernel_total_ms = stats["primary_total_ns"] / 1e6
                kernel_calls = stats["primary_calls"]
                top_5 = stats["top_5_kernels"]
            # Estimate rocprofv3 overhead: traced is N/500 of the steps, scale to compare.
            traced_wall = traced["wall_seconds"]
            scaled = traced_wall * (DEFAULT_E2E_NSTEPS / DEFAULT_PROF_NSTEPS)
            prof_overhead = max(0.0, 100.0 * (scaled - untraced_walltime) / max(untraced_walltime, 1e-6))

    peak_vram = _peak_vram_mb_in_container() or 0.0

    print_contract(
        correctness=correctness,
        kernel_avg_us=kernel_avg_us,
        kernel_total_ms=kernel_total_ms,
        kernel_calls=kernel_calls,
        e2e_throughput=e2e_thr,
        e2e_wall_seconds=e2e_wall,
        peak_vram_mb=peak_vram,
        profile_overhead_pct=prof_overhead,
        top_5=top_5,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
