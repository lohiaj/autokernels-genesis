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

Reference baseline + bench config: harness.toml. Project context: references/project_context.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

from _config import cfg

# ---------------------------------------------------------------------------
# Constants (sourced from harness.toml via _config.cfg)
# ---------------------------------------------------------------------------

H100_THROUGHPUT_REF = cfg.get_reference_value()       # back-compat alias
DEFAULT_E2E_NSTEPS  = cfg.get_default_e2e_nsteps()
DEFAULT_PROF_NSTEPS = cfg.get_default_prof_nsteps()
DEFAULT_NENVS       = cfg.get_default_n_envs()
DEFAULT_PRECISION   = cfg.get_default_precision()
DEFAULT_TRIALS      = cfg.get_default_trials()

UNTRACED_TIMEOUT_S    = cfg.get_untraced_timeout_s()
TRACED_TIMEOUT_S      = cfg.get_traced_timeout_s()
CORRECTNESS_TIMEOUT_S = cfg.get_correctness_timeout_s()

EXCLUDE_KERNEL_RE = re.compile(cfg.get_exclude_kernel_re())

# Campaign manifest path relative to repo root
SCRIPT_DIR = Path(__file__).resolve().parent
KERNELS_DIR = SCRIPT_DIR / "kernels"
WORKSPACE_DIR = SCRIPT_DIR / "workspace"
RUNS_DIR = SCRIPT_DIR / "runs"

# ---------------------------------------------------------------------------
# Forcing-function rubric enforcement
# ---------------------------------------------------------------------------
# program.md::B1.5 requires every hypothesis commit to include three lines:
#   1. Current dominant bottleneck:  ...
#   2. Smallest change to move it:   ...
#   3. Prior(working): 0.NN          -- ...
# This catches the "agent forgot the rubric and reverted to checklist mode"
# failure that produced the original 4-5-iteration plateau. Bypasses:
#   - commit message starting with "baseline" or "exp 0:" (initial calibration)
#   - --no-rubric-check on the bench.py command line
# Intentionally permissive on the third line (regex matches a probability or
# the literal "skip" tag) so explicit deviations can be logged without nagging.

RUBRIC_RE_BOTTLENECK = re.compile(r"^\s*1\.\s*(?:current\s+)?(?:dominant\s+)?bottleneck", re.I | re.M)
RUBRIC_RE_CHANGE     = re.compile(r"^\s*2\.\s*(?:smallest\s+)?change", re.I | re.M)
RUBRIC_RE_PRIOR      = re.compile(r"^\s*3\.\s*prior", re.I | re.M)
RUBRIC_BASELINE_RE   = re.compile(r"^\s*(baseline|exp\s*0\b)", re.I | re.M)


def _head_commit_msg() -> str:
    """Return the latest commit message body in $GENESIS_SRC, or '' on error."""
    src = _genesis_src()
    try:
        r = subprocess.run(
            ["git", "-C", src, "log", "-1", "--pretty=%B"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            return r.stdout or ""
    except (subprocess.SubprocessError, OSError):
        pass
    return ""


def _maybe_verify_sandbox() -> None:
    """Run `sandbox.py verify` if a session is active. Halts with exit 2 on tamper.
    Silent no-op if no session is active (e.g. manual bench outside a sandbox)."""
    sandbox_py = Path(__file__).resolve().parent / "sandbox.py"
    if not sandbox_py.exists():
        return  # sandbox.py removed?
    # Cheap check: does the session_state.json exist? If not, skip verify.
    sandbox_dir = os.environ.get("AUTOKERNEL_SANDBOX")
    if sandbox_dir:
        state = Path(sandbox_dir) / "session_state.json"
    else:
        workspace_root = os.environ.get("AUTOKERNEL_ROOT") or os.path.expanduser("~/work")
        state = Path(workspace_root) / ".cache" / "autokernels-genesis-sandbox" / "session_state.json"
    if not state.exists():
        return  # No active session; nothing to verify.

    rc = subprocess.call([sys.executable, str(sandbox_py), "verify"])
    if rc != 0:
        # sandbox.py verify already printed a clear error to stderr.
        die("sandbox tamper detected; refusing to bench. Re-run "
            "`uv run sandbox.py setup --kernel <K>` after recovering, or pass "
            "--no-sandbox-verify to bypass (NOT recommended).")


def check_rubric_or_die(skip: bool) -> None:
    """Enforce the B1.5 forcing-function rubric on the HEAD commit message.

    Skipped if `skip` is True or the commit subject is a baseline.
    Exits 1 with a structured error if the rubric is missing -- this is what
    converts the rubric from an aspirational instruction into structural
    enforcement, since the bench refuses to spend cycles on un-thought experiments.
    """
    if skip:
        return
    msg = _head_commit_msg()
    if not msg.strip():
        warn("could not read HEAD commit message in $GENESIS_SRC; rubric check skipped")
        return
    if RUBRIC_BASELINE_RE.search(msg):
        return  # baseline run -- rubric not required
    have = [
        ("1. bottleneck", bool(RUBRIC_RE_BOTTLENECK.search(msg))),
        ("2. change",     bool(RUBRIC_RE_CHANGE.search(msg))),
        ("3. prior",      bool(RUBRIC_RE_PRIOR.search(msg))),
    ]
    missing = [k for k, ok in have if not ok]
    if missing:
        print()
        print("=" * 72, file=sys.stderr)
        print("bench: ABORT -- HEAD commit message is missing the B1.5 rubric.", file=sys.stderr)
        print(f"missing lines: {missing}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Required (in the commit body, anywhere -- one per line):", file=sys.stderr)
        print("  1. Current dominant bottleneck:  <cited from rocprofv3 / omniperf>", file=sys.stderr)
        print("  2. Smallest change to move it:   <one file, one symbol, one construct>", file=sys.stderr)
        print("  3. Prior(working): 0.NN          -- <one sentence>", file=sys.stderr)
        print("", file=sys.stderr)
        print("Fix: amend the commit, OR pass --no-rubric-check (avoid; it bypasses", file=sys.stderr)
        print("     the forcing function that exists to prevent checklist-mode reverts).", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Container exec wrapper
# ---------------------------------------------------------------------------

def _container() -> str:
    name = os.environ.get("AUTOKERNEL_CONTAINER")
    if not name:
        gpu = os.environ.get("AUTOKERNEL_GPU_ID", "0")
        name = cfg.get_container_name(gpu)
    return name

def _genesis_src() -> str:
    return os.environ.get("GENESIS_SRC") or cfg.get_project_src_default()

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
    """Wipe project-specific kernel caches inside the container.

    Caches that survive an edit will silently no-op the patch (the JIT serves
    the previous compiled kernel). The list of paths to wipe is configured in
    harness.toml::container.caches_to_wipe.
    """
    paths = " ".join(shlex.quote(p) for p in cfg.get_caches_to_wipe())
    rc, out = _docker_exec(
        f"rm -rf {paths} 2>/dev/null; true",
        timeout=60,
    )
    if rc != 0:
        warn(f"cache wipe returned rc={rc}: {out[-300:]}")


# ---------------------------------------------------------------------------
# GPU clock pinning -- mitigates inter-trial variance from frequency scaling
# ---------------------------------------------------------------------------
# CDNA3 default perf level is "auto", which lets the GPU throttle based on
# thermals and load. Across 3-trial benches that's a meaningful source of the
# ~1% noise floor. Pinning to "high" before the bench and restoring after
# typically halves inter-trial sigma. Safe noop if rocm-smi isn't available
# (e.g., dev box without root inside container).

def pin_gpu_clocks() -> str | None:
    """Pin the agent's GPU to high perf level. Returns previous level for
    restore, or None if pinning failed (best-effort; commands are configurable
    via harness.toml::gpu)."""
    gpu = os.environ.get("AUTOKERNEL_GPU_ID", "0")
    rc, out = _docker_exec(f"{cfg.get_gpu_read_perf_command(gpu)} 2>&1 || true", timeout=15)
    prev = None
    if rc == 0:
        m = re.search(cfg.get_prev_level_regex(), out)
        if m:
            prev = m.group(1)
    rc2, out2 = _docker_exec(f"{cfg.get_gpu_pin_command(gpu)} 2>&1 || true", timeout=15)
    if rc2 == 0 and "ERROR" not in out2.upper() and "Permission" not in out2:
        print(f"bench: pinned GPU{gpu} clocks to 'high' (was '{prev or 'unknown'}')")
        return prev or "auto"
    warn(f"could not pin GPU{gpu} clocks (perf-control tool unavailable or no permission); "
         f"trial variance may be elevated")
    return None


def restore_gpu_clocks(prev: str | None) -> None:
    """Restore GPU perf level. No-op if pin_gpu_clocks failed."""
    if prev is None:
        return
    gpu = os.environ.get("AUTOKERNEL_GPU_ID", "0")
    _docker_exec(f"{cfg.get_gpu_restore_command(gpu, prev)} 2>&1 || true", timeout=15)
    print(f"bench: restored GPU{gpu} perf level to '{prev}'")


def run_correctness(manifest: dict) -> tuple[str, str]:
    """Run the scoped pytest subset. Returns (status, log_tail).

    cwd, env vars, and pytest flags are all configured in harness.toml::correctness.
    """
    tests = manifest["correctness_tests"]
    if not tests:
        return "SKIP", "no correctness_tests listed"
    test_args = " ".join(shlex.quote(t) for t in tests)
    cmd = (
        f"cd {cfg.get_pytest_dir()} && "
        f"{cfg.get_pytest_extra_env()} timeout {CORRECTNESS_TIMEOUT_S - 30} "
        f"python -m pytest {test_args} {cfg.get_pytest_args()} 2>&1 | tail -200"
    )
    rc, out = _docker_exec(cmd, timeout=CORRECTNESS_TIMEOUT_S)
    if rc == 124:
        return "TIMEOUT", out[-400:]
    if rc == 0:
        return "PASS", out[-400:]
    if "FAILED" in out or "ERROR" in out:
        return "FAIL", out[-1000:]
    return "FAIL", f"rc={rc}\n{out[-800:]}"


def _run_untraced_trial(out_dir: str, trial_idx: int) -> dict:
    """Run a SINGLE untraced trial. Returns dict with status + throughput/wall on PASS."""
    json_out = f"{out_dir}/untraced_t{trial_idx}.json"
    cmd = (
        f"mkdir -p {out_dir} && "
        f"{cfg.get_container_env_str()} "
        f"timeout {UNTRACED_TIMEOUT_S - 30} "
        f"python3 {cfg.get_bench_script()} "
        f"--precision {DEFAULT_PRECISION} "
        f"--n-envs {DEFAULT_NENVS} "
        f"--num-steps {DEFAULT_E2E_NSTEPS} "
        f"--tag untraced_t{trial_idx} "
        f"--out {json_out} 2>&1 | tail -30"
    )
    t0 = time.time()
    rc, out = _docker_exec(cmd, timeout=UNTRACED_TIMEOUT_S)
    dt = time.time() - t0

    if rc == 124:
        return {"status": "TIMEOUT", "wall_seconds": dt, "log_tail": out[-800:]}
    if rc != 0:
        return {"status": "CRASH", "wall_seconds": dt, "log_tail": out[-1500:]}

    parsed = _read_json_in_container(json_out)
    if parsed is None:
        return {"status": "PARSE_FAIL", "wall_seconds": dt, "log_tail": out[-800:]}

    entry = parsed if isinstance(parsed, dict) else _select_8192_entry(parsed)
    if entry is None:
        return {"status": "PARSE_FAIL", "wall_seconds": dt, "log_tail": out[-800:],
                "raw": str(parsed)[:400]}

    throughput = _first_present(entry, cfg.get_metric_json_keys()) or 0.0
    wall       = _first_present(entry, cfg.get_wall_json_keys()) or 0.0
    return {
        "status": "PASS",
        "e2e_throughput": float(throughput),
        "e2e_wall_seconds": float(wall),
        "wall_seconds": dt,
        "log_tail": out[-200:],
    }


def run_untraced_bench(out_dir: str, n_trials: int = DEFAULT_TRIALS,
                       wipe_between: bool = True) -> dict:
    """Run N untraced trials and report mean/sigma.

    Why N trials: at the documented ~1% noise floor, single-trial results
    routinely produce false KEEPs and false REVERTs. Mean over N>=3 with sigma
    enables a 2-sigma KEEP rule that actually distinguishes signal from noise.

    `wipe_between=True` clears the kernel cache before EACH trial so we measure
    cold-cache compile + steady-state separately. Set False to amortize compile.
    """
    samples: list[float] = []
    walls: list[float] = []
    total_wall = 0.0
    last_log_tail = ""
    last_status = "PASS"

    for i in range(max(1, n_trials)):
        if wipe_between and i > 0:
            # Re-wipe so each trial sees the same cold-cache scenario as trial 0
            wipe_caches()
        print(f"bench:   trial {i+1}/{n_trials}...")
        r = _run_untraced_trial(out_dir, i)
        total_wall += r.get("wall_seconds", 0.0)
        last_log_tail = r.get("log_tail", "")
        if r["status"] != "PASS":
            last_status = r["status"]
            # Abort multi-trial on first hard failure -- one bad trial means
            # the change is broken; no point spending budget on more.
            return {
                "status": r["status"],
                "wall_seconds": total_wall,
                "log_tail": last_log_tail,
                "trials_completed": i,
            }
        samples.append(r["e2e_throughput"])
        walls.append(r["e2e_wall_seconds"])
        print(f"bench:   trial {i+1}: e2e={r['e2e_throughput']:.0f} wall={r['e2e_wall_seconds']:.2f}s")

    mean = statistics.mean(samples)
    sigma = statistics.stdev(samples) if len(samples) >= 2 else 0.0
    sigma_pct = (100.0 * sigma / mean) if mean else 0.0
    wall_mean = statistics.mean(walls)

    return {
        "status": "PASS",
        "e2e_throughput": mean,
        "e2e_throughput_sigma": sigma,
        "e2e_throughput_sigma_pct": sigma_pct,
        "e2e_throughput_n": len(samples),
        "e2e_throughput_samples": samples,
        "e2e_wall_seconds": wall_mean,
        "wall_seconds": total_wall,
        "log_tail": last_log_tail,
    }


def run_traced_bench(out_dir: str) -> dict:
    """Run benchmark under the configured profiler for per-kernel attribution."""
    cmd = (
        f"mkdir -p {out_dir} && cd {out_dir} && "
        f"{cfg.get_container_env_str()} "
        f"timeout {TRACED_TIMEOUT_S - 60} "
        f"{cfg.get_profiler_command()} "
        f"python3 {cfg.get_bench_script()} "
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

    csv_name = "traced" + cfg.get_stats_csv_suffix()  # e.g. "traced_kernel_stats.csv"
    return {"status": "PASS", "wall_seconds": dt, "csv_path": f"{out_dir}/{csv_name}"}


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

def _first_present(d: dict, keys: list[str]) -> object | None:
    """Return the first non-falsy value from `d` for any of `keys`. Used for
    extracting metric/wall fields from bench-script JSON without hardcoding key
    names (different bench scripts use different conventions)."""
    for k in keys:
        v = d.get(k)
        if v:
            return v
    return None


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


# ---------------------------------------------------------------------------
# AMDGCN dump (Move 37 escape hatch)
# ---------------------------------------------------------------------------
# Extracts the LLVM IR that Quadrants caches in /root/.cache/quadrants/qdcache/.
# The IR is amdgcn-targeted -- it shows exactly what the compiler chose to
# emit for this kernel, including alloca/spill patterns, load/store address
# spaces, intrinsic usage, inlining decisions. It's the closest off-distribution
# input we can give the agent without recompiling Quadrants itself.
#
# We do NOT pre-analyze the dump (no regex pattern detector). The agent reads
# the IR directly and decides what's interesting. See references/amdgcn_patterns.md
# for examples of what to look for, but the agent is free to spot novel things.
#
# Returns a list of paths to .ll files inside the container.

def _dump_amdgcn_in_container(out_dir: str, marker_file: str) -> list[str]:
    """Find Quadrants cache files modified since marker_file was touched,
    extract LLVM IR via `strings`, save to out_dir/amdgcn/. Returns list of
    container-side paths to the produced .ll files (or [] if none / unsupported)."""
    cmd = (
        f"mkdir -p {out_dir}/amdgcn && "
        # Quadrants cache lives under ~/.cache/quadrants on the user inside the
        # container. The kernel-compilation-manager subdir holds .tic files, each
        # containing the LLVM IR for one compiled kernel.
        f"CACHE=/root/.cache/quadrants/qdcache/kernel_compilation_manager && "
        f"if [ ! -d \"$CACHE\" ]; then echo 'NO_CACHE'; exit 0; fi && "
        f"COUNT=0 && "
        # -newer reads the marker's mtime; -type f filters; -name only .tic
        f"for tic in $(find \"$CACHE\" -name '*.tic' -newer {marker_file} -type f 2>/dev/null); do "
        f"    name=$(basename \"$tic\" .tic | head -c 16); "
        f"    out={out_dir}/amdgcn/${{name}}.ll; "
        # `strings` extracts the readable text from the .tic serialization.
        # `sed` keeps everything from the LLVM `target triple` line onward
        # (drops the cache header noise).
        f"    strings \"$tic\" 2>/dev/null | sed -n '/^target triple/,$p' > \"$out\"; "
        f"    if [ -s \"$out\" ]; then echo \"$out\"; COUNT=$((COUNT+1)); else rm -f \"$out\"; fi; "
        f"done; "
        f"echo \"AMDGCN_DUMP_COUNT=$COUNT\" >&2"
    )
    rc, out = _docker_exec(cmd, timeout=60)
    if rc != 0:
        warn(f"amdgcn dump failed (rc={rc}): {out[-300:]}")
        return []
    if "NO_CACHE" in out:
        warn("amdgcn dump: Quadrants cache not found (non-Quadrants project? skipping)")
        return []
    paths = [line.strip() for line in out.splitlines()
             if line.strip().endswith(".ll")]
    return paths


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
                   top_5: list[tuple[str, float, float]],
                   e2e_throughput_sigma: float = 0.0,
                   e2e_throughput_sigma_pct: float = 0.0,
                   e2e_throughput_n: int = 1,
                   e2e_throughput_samples: list[float] | None = None) -> None:
    print()
    print("---")
    print(f"correctness:        {correctness}")
    print(f"kernel_avg_us:      {kernel_avg_us:.2f}")
    print(f"kernel_total_ms:    {kernel_total_ms:.2f}")
    print(f"kernel_calls:       {kernel_calls}")
    print(f"e2e_throughput:     {e2e_throughput:.0f}")
    print(f"e2e_throughput_sigma: {e2e_throughput_sigma:.1f}")
    print(f"e2e_throughput_sigma_pct: {e2e_throughput_sigma_pct:.3f}")
    print(f"e2e_throughput_n:   {e2e_throughput_n}")
    ref = cfg.get_reference_value()
    pct = (100.0 * e2e_throughput / ref) if ref else 0.0
    # Key is kept as e2e_pct_of_h100 for back-compat with existing parsers /
    # results.tsv schema; the value is "% of whatever reference_value is set to".
    print(f"e2e_pct_of_h100:    {pct:.2f}")
    print(f"e2e_wall_seconds:   {e2e_wall_seconds:.3f}")
    print(f"peak_vram_mb:       {peak_vram_mb:.1f}")
    vpct = (100.0 * peak_vram_mb / (192 * 1024)) if peak_vram_mb else 0.0
    print(f"peak_vram_pct:      {vpct:.1f}")
    print(f"profile_overhead_pct: {profile_overhead_pct:.1f}")
    if e2e_throughput_samples and len(e2e_throughput_samples) > 1:
        sample_str = ", ".join(f"{s:.0f}" for s in e2e_throughput_samples)
        print(f"e2e_throughput_samples: [{sample_str}]")
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
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                    help=f"untraced trials (default {DEFAULT_TRIALS}); >=2 enables sigma reporting")
    ap.add_argument("--no-wipe-between-trials", action="store_true",
                    help="reuse cached compile across trials (faster, but conflates compile+steady-state)")
    ap.add_argument("--no-rubric-check", action="store_true",
                    help="bypass the B1.5 three-question rubric enforcement on the HEAD commit "
                         "(use only for explicit out-of-band runs)")
    ap.add_argument("--dump-amdgcn", action="store_true",
                    help="extract LLVM IR for kernels compiled during this bench (Move 37 escape "
                         "hatch -- gives the agent off-distribution compiler output to reason about). "
                         "See references/amdgcn_patterns.md for what to look for.")
    ap.add_argument("--no-sandbox-verify", action="store_true",
                    help="skip the pre-bench sandbox tamper check (sandbox.py verify). "
                         "Use when running manually outside a sandbox session.")
    args = ap.parse_args()

    # Sanity
    container = _container()
    if not _container_exists(container):
        die(f"container {container} is not running. Did you run launcher/launch_8gpu.sh?")
    if not os.path.isdir(_genesis_src()):
        die(f"GENESIS_SRC={_genesis_src()} is not a directory")

    manifest = load_manifest(args.campaign)
    verify_edit_files_exist(manifest)

    # Forcing-function gate: refuse to bench an ill-formed hypothesis commit.
    check_rubric_or_die(skip=args.no_rubric_check)

    # Bug 3 fix: detect mid-session sandbox tampering before running the bench.
    # If the sandbox session_state.json exists, verify the expected branches are
    # still checked out. Skipped if no session state (manual bench, no sandbox).
    if not args.no_sandbox_verify:
        _maybe_verify_sandbox()

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

    # Make a per-experiment workspace dir inside the container.
    # runs_dir is configured in harness.toml::container.runs_dir.
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir_in_container = f"{cfg.get_runs_dir()}/{args.campaign}-{run_id}"

    # If the user wants the AMDGCN dump, drop a marker file BEFORE the bench so
    # we can find only the kernels compiled during THIS run (not stale cache).
    amdgcn_marker = None
    if args.dump_amdgcn:
        amdgcn_marker = f"{out_dir_in_container}/.amdgcn_marker"
        _docker_exec(f"mkdir -p {out_dir_in_container} && touch {amdgcn_marker}", timeout=15)

    # Step 3: untraced bench (headline e2e) -- N trials for variance estimate.
    # Pin GPU clocks for the duration of the WHOLE bench (untraced + traced) to
    # reduce inter-trial frequency-scaling noise. Use atexit so the restore
    # fires on every exit path (early return, exception, sys.exit) without
    # forcing us to re-indent the whole bench body in a try/finally.
    import atexit
    prev_perf = pin_gpu_clocks()
    atexit.register(restore_gpu_clocks, prev_perf)

    print(f"bench: untraced run x{args.trials} (8192/500/FP32)...")
    untraced = run_untraced_bench(
        out_dir_in_container,
        n_trials=args.trials,
        wipe_between=not args.no_wipe_between_trials,
    )
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
    e2e_thr_sigma = untraced.get("e2e_throughput_sigma", 0.0)
    e2e_thr_sigma_pct = untraced.get("e2e_throughput_sigma_pct", 0.0)
    e2e_thr_n = untraced.get("e2e_throughput_n", 1)
    e2e_thr_samples = untraced.get("e2e_throughput_samples", [e2e_thr])
    e2e_wall = untraced["e2e_wall_seconds"]
    untraced_walltime = untraced["wall_seconds"]
    print(f"bench: untraced e2e mean={e2e_thr:.0f} sigma={e2e_thr_sigma:.0f} ({e2e_thr_sigma_pct:.2f}%) "
          f"n={e2e_thr_n} wall_total={untraced_walltime:.1f}s")

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
            # Estimate rocprofv3 overhead: traced is N/500 of the steps, scale to compare
            # against the *per-trial* untraced wall (not the multi-trial sum).
            traced_wall = traced["wall_seconds"]
            scaled = traced_wall * (DEFAULT_E2E_NSTEPS / DEFAULT_PROF_NSTEPS)
            untraced_per_trial = untraced_walltime / max(e2e_thr_n, 1)
            prof_overhead = max(
                0.0,
                100.0 * (scaled - untraced_per_trial) / max(untraced_per_trial, 1e-6),
            )

    peak_vram = _peak_vram_mb_in_container() or 0.0

    # AMDGCN dump (Move 37 escape hatch) -- emit AFTER bench so the cache is
    # populated. Print paths to stderr so the agent can find them.
    if amdgcn_marker:
        print("bench: dumping AMDGCN/LLVM IR for kernels compiled this run...")
        ll_paths = _dump_amdgcn_in_container(out_dir_in_container, amdgcn_marker)
        if ll_paths:
            print(f"amdgcn dump: {len(ll_paths)} kernel(s) extracted to {out_dir_in_container}/amdgcn/",
                  file=sys.stderr)
            for p in ll_paths:
                print(f"amdgcn dump: {p}", file=sys.stderr)
        else:
            print("amdgcn dump: no kernels extracted (cache empty or non-Quadrants project)",
                  file=sys.stderr)

    print_contract(
        correctness=correctness,
        kernel_avg_us=kernel_avg_us,
        kernel_total_ms=kernel_total_ms,
        kernel_calls=kernel_calls,
        e2e_throughput=e2e_thr,
        e2e_throughput_sigma=e2e_thr_sigma,
        e2e_throughput_sigma_pct=e2e_thr_sigma_pct,
        e2e_throughput_n=e2e_thr_n,
        e2e_throughput_samples=e2e_thr_samples,
        e2e_wall_seconds=e2e_wall,
        peak_vram_mb=peak_vram,
        profile_overhead_pct=prof_overhead,
        top_5=top_5,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
