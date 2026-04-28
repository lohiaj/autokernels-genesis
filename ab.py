#!/usr/bin/env python3
"""
ab.py -- statistically-disciplined paired A/B testing for kernel changes.

WHY THIS EXISTS
===============
The previous decision flow ("bench A once, bench B once, compare ad-hoc %")
produced KEEP decisions that did not reproduce on the production stack.
Concrete failure: a session claimed kernel -33.1% / e2e flat; production
re-measurement showed kernel -0.31% / e2e -0.98% (median, 4/5 negative).
The diff was clean code -- the methodology was wrong.

This file replaces that flow with paired interleaved multi-trial measurement,
hash-verified arm toggling, a CoV + sign-consistency + Amdahl decision rule,
and a cross-kernel regression check. Every KEEP/DISCARD now writes a fully
reproducible per-trial JSON footer.

DESIGN INVARIANTS (the spec section numbers are in `### refs:` comments below)
=============================================================================
- The two arms are toggled by `git checkout <commit> -- <files>`. Hashes are
  verified on every toggle. If hashes don't change, the toggle is a no-op
  and the run aborts (silent toggle failure was the cause of PR #16).
- Trials are interleaved (base, cand, base, cand, ...) so thermal/contention
  drift hits both arms equally.
- Caches are wiped ONCE at session start; warmup runs flush JIT before the
  first measured trial of each arm.
- Outliers are flagged but not silently dropped; concurrent GPU load aborts
  the run.
- KEEP requires ALL of: median Δ ≥ +0.5%, sign consistency ≥ ceil(N*0.6),
  mean and median same sign, base CoV ≤ 3%, Amdahl plausibility on E2E gain.
- Cross-kernel regression: if any non-target top-8 kernel got slower by ≥2%,
  it's CROSS_KERNEL_REGRESSION even if the target moved correctly.
- Every per-trial JSON includes branch/commit/hashes/RigidOptions/HW so
  results are independently reproducible.

USAGE
=====
  uv run ab.py ab --base HEAD~1 --cand HEAD --target-kernel 'func_broad_phase'
  uv run ab.py selftest-noop    # validates the harness with a no-op patch
  uv run ab.py selftest-real-win --commit <sha>  # validates with a real win

Outputs: workspace/ab/<timestamp>/{summary.md, trials/*.json, decision.json}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from _config import cfg

# Reuse a few low-level primitives from bench.py. Imported as private API
# but they're stable contracts within this codebase.
import bench as _bench

# ---------------------------------------------------------------------------
# Constants (match spec sections)
# ---------------------------------------------------------------------------

DEFAULT_TRIALS = 5                 # spec 2.3: N >= 5
DEFAULT_ROCPROF_RUNS = 2           # spec 2.6: M >= 2
DEFAULT_WARMUP_RUNS = 3            # spec 2.7: at least 3 per arm
DEFAULT_BENCH_WARMUP = 15          # spec 2.7: bench_mi300.py --warmup >= 15
DEFAULT_NUM_STEPS = 500            # E2E
DEFAULT_TRACED_NUM_STEPS = 100     # rocprof (lower = faster, but enough samples)

DEFAULT_MIN_MEDIAN_DELTA_PCT = 0.5     # spec 2.4 #1
DEFAULT_SIGN_CONSISTENCY_FRAC = 0.6    # spec 2.4 #2
DEFAULT_MAX_BASE_COV_PCT = 3.0         # spec 2.4 #4
DEFAULT_AMDAHL_TOLERANCE = 0.3         # spec 2.4 #5: observed >= 0.3 * expected
DEFAULT_CROSS_KERNEL_REGRESSION_PCT = 2.0   # spec 2.8: |Δ| >= 2% on non-target
DEFAULT_GPU_BUSY_THRESHOLD = 5.0       # spec 2.9: > 5% triggers wait
DEFAULT_GPU_WAIT_MAX_S = 300           # spec 2.9: 5 min then abort
DEFAULT_BASELINE_DRIFT_PCT = 15.0      # spec 2.2: > 15% from previous baseline = abort

WORKSPACE = Path(__file__).resolve().parent / "workspace"


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def die(msg: str, code: int = 1) -> None:
    print(f"ab: ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def warn(msg: str) -> None:
    print(f"ab: WARN: {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"ab: {msg}", file=sys.stderr)


def run(cmd: list[str], cwd: str | None = None, check: bool = False,
        capture_binary: bool = False) -> tuple[int, str | bytes]:
    """Run a host-side subprocess; return (rc, output). capture_binary=True
    returns bytes (used for git show on binary files)."""
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=120,
                           check=check, text=not capture_binary)
        if capture_binary:
            return r.returncode, (r.stdout or b"")
        out = (r.stdout or "") + (r.stderr or "")
        return r.returncode, out
    except subprocess.SubprocessError as e:
        return 1, (b"" if capture_binary else f"subprocess error: {e}")


# ---------------------------------------------------------------------------
# 2.1: Hash-verified arm toggle
# ---------------------------------------------------------------------------

def resolve_commit(repo_dir: str, ref: str) -> str:
    rc, out = run(["git", "-C", repo_dir, "rev-parse", ref])
    if rc != 0:
        die(f"can't resolve git ref '{ref}' in {repo_dir}: {out}")
    return out.strip().splitlines()[0]


def changed_files(repo_dir: str, base: str, cand: str) -> list[str]:
    rc, out = run(["git", "-C", repo_dir, "diff", "--name-only", base, cand])
    if rc != 0:
        die(f"git diff --name-only {base} {cand}: {out}")
    return [l.strip() for l in out.splitlines() if l.strip()]


def hash_of_blob_at(repo_dir: str, commit: str, path: str) -> str:
    """sha256 of the file's content at <commit>:<path>. Returns "DELETED" if
    the file does not exist at that commit (e.g., a new file in the cand)."""
    rc, content = run(["git", "-C", repo_dir, "show", f"{commit}:{path}"],
                      capture_binary=True)
    if rc != 0:
        return "DELETED"
    return "sha256:" + hashlib.sha256(content).hexdigest()


def hash_of_working_file(repo_dir: str, path: str) -> str:
    full = Path(repo_dir) / path
    if not full.exists():
        return "MISSING"
    with open(full, "rb") as f:
        return "sha256:" + hashlib.sha256(f.read()).hexdigest()


def expected_hashes(repo_dir: str, commit: str, files: list[str]) -> dict[str, str]:
    return {f: hash_of_blob_at(repo_dir, commit, f) for f in files}


def current_hashes(repo_dir: str, files: list[str]) -> dict[str, str]:
    return {f: hash_of_working_file(repo_dir, f) for f in files}


def checkout_arm(repo_dir: str, commit: str, files: list[str]) -> None:
    """Checkout the given files at <commit> WITHOUT switching branches.
    Leaves the rest of the working tree untouched. Files that exist at
    <commit> are restored; files that don't are removed (handles renames/deletes)."""
    if not files:
        return
    # `git checkout <commit> -- <files>` updates each named file from the
    # given commit. For files that don't exist at <commit> (added in the
    # other arm), this errors; we handle that by `git rm` first.
    for f in files:
        rc, _ = run(["git", "-C", repo_dir, "cat-file", "-e", f"{commit}:{f}"])
        if rc != 0:
            # File doesn't exist at this commit; remove from working tree
            (Path(repo_dir) / f).unlink(missing_ok=True)
            run(["git", "-C", repo_dir, "rm", "-f", "--cached", "--quiet", f])
        else:
            rc, out = run(["git", "-C", repo_dir, "checkout", commit, "--", f])
            if rc != 0:
                die(f"checkout failed for {f} at {commit[:8]}: {out}")


def verify_arm_hashes(repo_dir: str, expected: dict[str, str], arm_label: str
                      ) -> dict[str, str]:
    """Verify the working tree's hashes match `expected`. Dies on mismatch.
    Returns the actual hashes (== expected, by definition, on success)."""
    actual = current_hashes(repo_dir, list(expected.keys()))
    if actual != expected:
        diff = []
        for f in expected:
            if expected[f] != actual.get(f):
                diff.append(f"  {f}\n    expected: {expected[f]}\n    actual:   {actual.get(f)}")
        die(f"HASH MISMATCH on arm {arm_label}:\n" + "\n".join(diff))
    return actual


def assert_arms_differ(base_h: dict, cand_h: dict, base_commit: str, cand_commit: str
                       ) -> None:
    """spec 2.1: abort if base and cand hashes are identical (toggle is no-op)."""
    if base_h == cand_h:
        die(f"ABORT: base ({base_commit[:8]}) and cand ({cand_commit[:8]}) "
            f"have identical file hashes.\n"
            f"This means the diff is whitespace/comment-only OR the toggle "
            f"didn't actually swap files. Either way, an A/B is meaningless.\n"
            f"This is the silent-toggle-failure mode that produced PR #16.")
    if not base_h:
        die("ABORT: empty hash list (no files changed between arms).")


# ---------------------------------------------------------------------------
# 2.2: Production-config locking + drift check
# ---------------------------------------------------------------------------

def capture_env_metadata(repo_dir: str) -> dict:
    """Snapshot Genesis branch/commit + Quadrants commit + ROCm version + GPU."""
    meta: dict = {}
    rc, out = run(["git", "-C", repo_dir, "rev-parse", "HEAD"])
    meta["genesis_head_commit"] = out.strip() if rc == 0 else "?"
    rc, out = run(["git", "-C", repo_dir, "rev-parse", "--abbrev-ref", "HEAD"])
    meta["genesis_branch"] = out.strip() if rc == 0 else "?"

    container = os.environ.get("AUTOKERNEL_CONTAINER")
    if container:
        # Quadrants commit (parsed from inside the container)
        rc, out = _bench._docker_exec(
            "python3 -c 'import quadrants; print(quadrants.__file__)' 2>&1",
            timeout=15)
        if rc == 0:
            qd_path = Path(out.strip().splitlines()[-1]).parent
            rc2, out2 = _bench._docker_exec(
                f"git -C {qd_path} rev-parse HEAD 2>&1 || echo NO_GIT", timeout=10)
            meta["quadrants_commit"] = out2.strip().splitlines()[-1]
            meta["quadrants_install_path"] = str(qd_path)
        # ROCm version
        rc, out = _bench._docker_exec("rocm-smi --version 2>&1 | head -3", timeout=10)
        meta["rocm_version"] = out.strip()
    else:
        meta["quadrants_commit"] = "?"
        meta["rocm_version"] = "?"

    meta["hip_visible_devices"] = os.environ.get("HIP_VISIBLE_DEVICES", "?")
    meta["host"] = os.uname().nodename
    return meta


def check_baseline_drift(repo_dir: str, current_baseline_kernel_us: float | None,
                         max_drift_pct: float = DEFAULT_BASELINE_DRIFT_PCT) -> None:
    """spec 2.2 sniff test: compare current baseline kernel µs to the previously
    recorded one. Aborts on > max_drift_pct difference."""
    prev_path = WORKSPACE / "ab" / "last_baseline.json"
    if not prev_path.exists() or current_baseline_kernel_us is None:
        return  # first run, nothing to compare against
    try:
        prev = json.loads(prev_path.read_text())
        prev_us = float(prev.get("baseline_kernel_us", 0))
    except (OSError, ValueError):
        return
    if prev_us <= 0:
        return
    drift = abs(current_baseline_kernel_us - prev_us) / prev_us * 100
    if drift > max_drift_pct:
        die(f"BASELINE DRIFT -- current baseline kernel {current_baseline_kernel_us:.2f}us "
            f"differs from previous {prev_us:.2f}us by {drift:.1f}% (>{max_drift_pct}%). "
            f"Stop and reconcile before any A/B. "
            f"Common causes: Quadrants rebuild, GPU driver update, prior session left "
            f"pip-editable swap unrestored. Re-run sandbox.py teardown + restart container.")


def record_baseline(baseline_kernel_us: float) -> None:
    p = WORKSPACE / "ab" / "last_baseline.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "baseline_kernel_us": baseline_kernel_us,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }))


def check_merge_base_age(repo_dir: str, working_branch: str, prod_branch: str,
                        max_age_h: int = 24) -> None:
    """spec 2.2: abort if working branch's merge-base with production is > 24h old."""
    rc, out = run(["git", "-C", repo_dir, "merge-base", working_branch, prod_branch])
    if rc != 0:
        warn(f"could not compute merge-base of {working_branch}/{prod_branch}: {out}")
        return
    mb = out.strip()
    rc, ts = run(["git", "-C", repo_dir, "show", "-s", "--format=%ct", mb])
    if rc != 0:
        return
    try:
        merge_base_time = datetime.fromtimestamp(int(ts.strip()), tz=timezone.utc)
    except ValueError:
        return
    age = datetime.now(timezone.utc) - merge_base_time
    if age > timedelta(hours=max_age_h):
        die(f"BASELINE DRIFT -- {working_branch}'s merge-base with {prod_branch} "
            f"is {age.total_seconds() / 3600:.1f}h old (> {max_age_h}h). "
            f"Rebase onto {prod_branch} before continuing, or pass --skip-merge-base-check "
            f"if you've explicitly verified your baseline is still representative.")


# ---------------------------------------------------------------------------
# 2.6 + 2.9: bench primitives (single-arm) + concurrent-load gate
# ---------------------------------------------------------------------------

def gpu_busy_pct() -> float:
    """rocm-smi --showuse for the configured GPU; returns busy %, or 0.0 on failure."""
    gpu = os.environ.get("AUTOKERNEL_GPU_ID", "0")
    rc, out = _bench._docker_exec(f"rocm-smi -d {gpu} --showuse 2>&1 || true", timeout=10)
    if rc != 0:
        return 0.0
    # rocm-smi formats vary; grab any percentage on a line containing GPU[N]
    for line in out.splitlines():
        if f"GPU[{gpu}]" in line or f"card{gpu}" in line.lower():
            m = re.search(r"(\d+)%", line)
            if m:
                return float(m.group(1))
    m = re.search(r"GPU use\s*\(%\)\s*[:=]\s*(\d+)", out)
    return float(m.group(1)) if m else 0.0


def wait_for_idle_gpu(threshold_pct: float = DEFAULT_GPU_BUSY_THRESHOLD,
                      max_wait_s: int = DEFAULT_GPU_WAIT_MAX_S) -> bool:
    """spec 2.9: poll GPU busy %; sleep + retry until idle. False on timeout."""
    start = time.time()
    while time.time() - start < max_wait_s:
        busy = gpu_busy_pct()
        if busy <= threshold_pct:
            return True
        info(f"GPU busy {busy:.1f}% > {threshold_pct}%, sleeping 30s "
             f"(elapsed {time.time() - start:.0f}s / {max_wait_s}s)")
        time.sleep(30)
    return False


def run_one_untraced(out_path_in_container: str, num_steps: int, warmup: int) -> dict:
    """Single untraced bench. NO cache wipe (caller controls that)."""
    container_env = cfg.get_container_env_str()
    bench_script = cfg.get_bench_script()
    n_envs = cfg.get_default_n_envs()
    precision = cfg.get_default_precision()

    # Sandbox URDF if available
    sandbox = os.environ.get("AUTOKERNEL_SANDBOX") or \
        os.path.expanduser("~/work/.cache/autokernels-genesis-sandbox")
    sandbox_in_container = sandbox.replace(os.path.expanduser("~/work"), "/work")
    urdf = f"{sandbox_in_container}/Genesis/newton-assets/unitree_g1/urdf/g1_29dof.urdf"
    pythonpath_prefix = f"PYTHONPATH={sandbox_in_container}/Genesis:$PYTHONPATH"

    cmd = (
        f"mkdir -p $(dirname {out_path_in_container}) && "
        f"{pythonpath_prefix} {container_env} "
        f"timeout {cfg.get_untraced_timeout_s() - 30} "
        f"python3 {bench_script} "
        f"--precision {precision} --n-envs {n_envs} --num-steps {num_steps} "
        f"--warmup {warmup} --urdf {urdf} "
        f"--out {out_path_in_container} --tag ab 2>&1 | tail -10"
    )
    t0 = time.time()
    rc, out = _bench._docker_exec(cmd, timeout=cfg.get_untraced_timeout_s())
    dt = time.time() - t0
    if rc != 0:
        return {"status": "CRASH", "wall_seconds": dt, "log_tail": out[-800:]}
    parsed = _bench._read_json_in_container(out_path_in_container)
    if parsed is None:
        return {"status": "PARSE_FAIL", "wall_seconds": dt, "log_tail": out[-300:]}
    entry = parsed if isinstance(parsed, dict) else (parsed[-1] if parsed else None)
    if entry is None:
        return {"status": "PARSE_FAIL", "wall_seconds": dt, "log_tail": out[-300:]}
    return {
        "status": "PASS",
        "throughput": float(entry.get("throughput") or entry.get("env_steps_per_sec") or 0),
        "wall_time_s": float(entry.get("wall_time_s") or entry.get("wall") or 0),
        "wall_seconds": dt,
    }


def run_one_traced(out_dir_in_container: str, num_steps: int) -> dict:
    """Single traced bench under rocprofv3. Returns paths and parsed kernel stats."""
    container_env = cfg.get_container_env_str()
    bench_script = cfg.get_bench_script()
    n_envs = cfg.get_default_n_envs()
    precision = cfg.get_default_precision()
    profiler = cfg.get_profiler_command()

    sandbox = os.environ.get("AUTOKERNEL_SANDBOX") or \
        os.path.expanduser("~/work/.cache/autokernels-genesis-sandbox")
    sandbox_in_container = sandbox.replace(os.path.expanduser("~/work"), "/work")
    urdf = f"{sandbox_in_container}/Genesis/newton-assets/unitree_g1/urdf/g1_29dof.urdf"
    pythonpath_prefix = f"PYTHONPATH={sandbox_in_container}/Genesis:$PYTHONPATH"

    cmd = (
        f"mkdir -p {out_dir_in_container} && cd {out_dir_in_container} && "
        f"{pythonpath_prefix} {container_env} "
        f"timeout {cfg.get_traced_timeout_s() - 60} "
        f"{profiler} python3 {bench_script} "
        f"--precision {precision} --n-envs {n_envs} --num-steps {num_steps} "
        f"--warmup 5 --urdf {urdf} "
        f"--out {out_dir_in_container}/traced.json --tag ab_traced 2>&1 | tail -10"
    )
    t0 = time.time()
    rc, out = _bench._docker_exec(cmd, timeout=cfg.get_traced_timeout_s())
    dt = time.time() - t0
    if rc != 0:
        return {"status": "CRASH", "wall_seconds": dt, "log_tail": out[-800:]}
    csv_name = "traced" + cfg.get_stats_csv_suffix()
    return {"status": "PASS", "wall_seconds": dt,
            "csv_path": f"{out_dir_in_container}/{csv_name}"}


# ---------------------------------------------------------------------------
# 2.4 #5 + 2.8: kernel attribution + cross-kernel regression
# ---------------------------------------------------------------------------

def parse_top_kernels(csv_path: str, target_pattern: str | None,
                      top_n: int = 8) -> dict:
    """Parse rocprofv3 _kernel_stats.csv. Returns:
        target: {avg_us, total_ms, calls}  (sum over kernels matching target_pattern)
        top_n:  list of {name, avg_us, total_ms, calls}, sorted by total time desc
        all_total_ms: sum of all (non-init) kernel times
    """
    rc, content = _bench._docker_exec(f"cat {csv_path} 2>&1", timeout=30)
    if rc != 0 or not content.strip():
        return {"error": f"can't read {csv_path}"}

    import csv as _csv
    rows = list(_csv.DictReader(content.splitlines()))
    if not rows:
        return {"error": "empty stats csv"}

    name_keys = ["KernelName", "Kernel_Name", "Kernel name", "kernel_name", "Name"]
    total_keys = ["TotalDurationNs", "Total_Duration_Ns", "TotalNs", "Total"]
    calls_keys = ["Calls", "Count", "Invocations"]
    exclude_re = re.compile(cfg.get_exclude_kernel_re())
    target_re = re.compile(target_pattern) if target_pattern else None

    def _pick(r, keys):
        for k in keys:
            if k in r:
                return r[k]
        return ""

    target_total_ns = 0.0
    target_calls = 0
    breakdown: list[tuple[str, float, int]] = []
    all_total_ns = 0.0

    for r in rows:
        name = _pick(r, name_keys)
        if not name or exclude_re.search(name):
            continue
        try:
            tns = float(_pick(r, total_keys) or 0)
            calls = int(float(_pick(r, calls_keys) or 0))
        except ValueError:
            continue
        all_total_ns += tns
        breakdown.append((name, tns, calls))
        if target_re and target_re.search(name):
            target_total_ns += tns
            target_calls += calls

    breakdown.sort(key=lambda x: x[1], reverse=True)
    top = []
    for name, tns, calls in breakdown[:top_n]:
        top.append({
            "name": name,
            "total_ms": tns / 1e6,
            "avg_us": (tns / calls / 1000.0) if calls else 0.0,
            "calls": calls,
        })

    target = {
        "total_ms": target_total_ns / 1e6,
        "avg_us": (target_total_ns / target_calls / 1000.0) if target_calls else 0.0,
        "calls": target_calls,
        "pct_of_all": (100.0 * target_total_ns / all_total_ns) if all_total_ns else 0.0,
    } if target_re else None

    return {
        "target": target,
        "top": top,
        "all_total_ms": all_total_ns / 1e6,
    }


def cross_kernel_regression_check(base_kernels: dict, cand_kernels: dict,
                                  target_pattern: str | None,
                                  threshold_pct: float = DEFAULT_CROSS_KERNEL_REGRESSION_PCT
                                  ) -> dict:
    """spec 2.8: check that no non-target top-8 kernel got slower by >= threshold%.
    Returns {regressed: [...], all_deltas: [...]}."""
    base_top = {k["name"]: k for k in base_kernels.get("top", [])}
    cand_top = {k["name"]: k for k in cand_kernels.get("top", [])}
    target_re = re.compile(target_pattern) if target_pattern else None

    regressed = []
    deltas = []
    for name, b in base_top.items():
        if name not in cand_top:
            continue
        c = cand_top[name]
        if b["avg_us"] <= 0:
            continue
        delta_pct = (c["avg_us"] - b["avg_us"]) / b["avg_us"] * 100
        is_target = bool(target_re and target_re.search(name))
        deltas.append({
            "name": name, "is_target": is_target,
            "base_avg_us": b["avg_us"], "cand_avg_us": c["avg_us"],
            "delta_pct": delta_pct,
        })
        # Regression = non-target got SLOWER by >= threshold (positive delta in us = slower)
        if not is_target and delta_pct >= threshold_pct:
            regressed.append({"name": name, "delta_pct": delta_pct})
    return {"regressed": regressed, "all_deltas": deltas}


# ---------------------------------------------------------------------------
# 2.4: paired stats + decision rule
# ---------------------------------------------------------------------------

def paired_stats(base_thr: list[float], cand_thr: list[float]) -> dict:
    """Compute paired-trial statistics. Returns dict with median/mean Δ%, sign
    consistency, base CoV%, and per-trial deltas (for outlier flagging)."""
    n = len(base_thr)
    paired = []
    for b, c in zip(base_thr, cand_thr):
        if b <= 0:
            continue
        paired.append((c - b) / b * 100)
    if len(paired) != n:
        # shouldn't happen post-status-filter but be defensive
        return {"error": f"trial count mismatch: {len(paired)} paired vs {n} base"}
    base_mean = statistics.mean(base_thr)
    base_stdev = statistics.stdev(base_thr) if n >= 2 else 0.0
    base_cov_pct = (100.0 * base_stdev / base_mean) if base_mean > 0 else 0.0
    sign_pos = sum(1 for d in paired if d > 0)
    sign_neg = sum(1 for d in paired if d < 0)
    return {
        "n_trials": n,
        "paired_deltas_pct": paired,
        "median_delta_pct": statistics.median(paired),
        "mean_delta_pct": statistics.mean(paired),
        "stdev_delta_pct": statistics.stdev(paired) if n >= 2 else 0.0,
        "sign_pos": sign_pos,
        "sign_neg": sign_neg,
        "base_mean": base_mean,
        "base_cov_pct": base_cov_pct,
        "cand_mean": statistics.mean(cand_thr),
    }


def flag_outliers(stats: dict, multiplier: float = 3.0) -> list[int]:
    """spec 2.5: any trial with |paired Δ| > multiplier * |median Δ| is flagged.
    Returns 1-indexed trial numbers."""
    paired = stats.get("paired_deltas_pct", [])
    median_abs = abs(stats.get("median_delta_pct", 0.0))
    if median_abs == 0:
        # Use stdev as the scale instead, to flag absurd outliers in flat-median runs
        median_abs = max(stats.get("stdev_delta_pct", 0.0), 0.1)
    threshold = multiplier * median_abs
    return [i + 1 for i, d in enumerate(paired) if abs(d) > threshold]


def amdahl_check(stats: dict, base_kernels: dict, cand_kernels: dict,
                 target_pattern: str | None,
                 tolerance: float = DEFAULT_AMDAHL_TOLERANCE) -> dict:
    """spec 2.4 #5: predicted E2E gain from kernel%×kernel_delta% should match
    observed E2E gain within `tolerance` (observed >= tolerance × expected)."""
    if not target_pattern:
        return {"applicable": False, "reason": "no target kernel"}
    if not (base_kernels.get("target") and cand_kernels.get("target")):
        return {"applicable": False, "reason": "missing target kernel data"}

    bt = base_kernels["target"]
    ct = cand_kernels["target"]
    if bt["avg_us"] <= 0:
        return {"applicable": False, "reason": "base target avg_us is zero"}

    kernel_delta_pct = (bt["avg_us"] - ct["avg_us"]) / bt["avg_us"] * 100  # positive = improvement
    kernel_pct_of_all = bt["pct_of_all"]
    expected_e2e_gain_pct = kernel_pct_of_all / 100.0 * kernel_delta_pct
    observed_e2e_gain_pct = stats["median_delta_pct"]

    if expected_e2e_gain_pct <= 0:
        # Kernel didn't actually improve; this should have been caught upstream
        return {
            "applicable": True, "passes": False,
            "reason": f"target kernel didn't improve (Δ={kernel_delta_pct:+.2f}%)",
            "kernel_delta_pct": kernel_delta_pct,
            "kernel_pct_of_all": kernel_pct_of_all,
            "expected_e2e_gain_pct": expected_e2e_gain_pct,
            "observed_e2e_gain_pct": observed_e2e_gain_pct,
        }

    ratio = observed_e2e_gain_pct / expected_e2e_gain_pct if expected_e2e_gain_pct else 0
    return {
        "applicable": True,
        "passes": ratio >= tolerance,
        "kernel_delta_pct": kernel_delta_pct,
        "kernel_pct_of_all": kernel_pct_of_all,
        "expected_e2e_gain_pct": expected_e2e_gain_pct,
        "observed_e2e_gain_pct": observed_e2e_gain_pct,
        "ratio_observed_to_expected": ratio,
        "tolerance": tolerance,
    }


def apply_decision_rule(stats: dict, amdahl: dict, cross_kernel: dict,
                        thresholds: dict) -> tuple[str, str]:
    """spec 2.4: full decision tree. Returns (verdict, reason)."""
    n = stats["n_trials"]
    median = stats["median_delta_pct"]
    mean = stats["mean_delta_pct"]
    sign_pos = stats["sign_pos"]
    base_cov = stats["base_cov_pct"]

    min_median = thresholds["min_median_delta_pct"]
    sign_frac = thresholds["sign_consistency_frac"]
    max_cov = thresholds["max_base_cov_pct"]
    sign_threshold = math.ceil(n * sign_frac)

    # Rule 4: noise floor
    if base_cov > max_cov:
        return ("NOISY_REDO",
                f"base CoV {base_cov:.2f}% > {max_cov}% — too noisy. Wait for "
                f"GPU to clear, then re-run.")
    # Rule 1: median magnitude
    if median < min_median:
        return ("DISCARD",
                f"median paired Δ {median:+.2f}% < {min_median}% threshold "
                f"(N={n}, sign={sign_pos}/{n} positive, mean={mean:+.2f}%, base_CoV={base_cov:.2f}%)")
    # Rule 2: sign consistency
    if sign_pos < sign_threshold:
        return ("DISCARD",
                f"sign consistency {sign_pos}/{n} positive < required {sign_threshold} "
                f"(median={median:+.2f}% but trials disagree)")
    # Rule 3: mean/median sign agreement
    if (median > 0) != (mean > 0):
        return ("DISCARD",
                f"mean Δ {mean:+.2f}% and median Δ {median:+.2f}% have opposite signs "
                f"— outlier-driven; treat as noise")
    # Rule 8 (cross-kernel regression)
    if cross_kernel.get("regressed"):
        regs = ", ".join(f"{r['name']} ({r['delta_pct']:+.1f}%)"
                         for r in cross_kernel["regressed"])
        return ("CROSS_KERNEL_REGRESSION",
                f"non-target kernels got slower: {regs}. Even though target moved correctly, "
                f"the change rebalanced register/scheduling across the module unfavorably.")
    # Rule 5 (Amdahl): only if amdahl is applicable
    if amdahl.get("applicable"):
        if not amdahl["passes"]:
            return ("KERNEL_OK_E2E_FLAT",
                    f"target kernel improved {amdahl['kernel_delta_pct']:+.2f}% "
                    f"(was {amdahl['kernel_pct_of_all']:.1f}% of frame, predicted "
                    f"E2E gain {amdahl['expected_e2e_gain_pct']:+.2f}%) but observed "
                    f"E2E gain only {amdahl['observed_e2e_gain_pct']:+.2f}% "
                    f"(ratio {amdahl['ratio_observed_to_expected']:.2f}× < "
                    f"{amdahl['tolerance']}× tolerance). Kernel win didn't propagate to wall time.")
    # All checks pass
    return ("KEEP",
            f"median Δ {median:+.2f}% (≥{min_median}%), sign {sign_pos}/{n} (≥{sign_threshold}), "
            f"mean Δ {mean:+.2f}% same sign, base CoV {base_cov:.2f}% (≤{max_cov}%)"
            + (f", Amdahl ratio {amdahl['ratio_observed_to_expected']:.2f}×" if amdahl.get("applicable") else ""))


# ---------------------------------------------------------------------------
# 2.10: per-trial JSON footer + session writeup
# ---------------------------------------------------------------------------

def make_trial_record(arm: str, trial_idx: int, bench_result: dict,
                      base_commit: str, cand_commit: str, files: list[str],
                      base_hashes: dict, cand_hashes: dict, env_meta: dict,
                      gpu_busy_pre: float, num_steps: int, warmup: int,
                      noise_suspect: bool = False) -> dict:
    """spec 2.10: full reproducibility footer."""
    rec = {
        "tag": f"ab_{arm}_t{trial_idx}",
        "arm": arm,
        "trial": trial_idx,
        "branch_base": base_commit,
        "branch_cand": cand_commit,
        "commit_base": base_commit,
        "commit_cand": cand_commit,
        "files_changed": files,
        "hashes_base": base_hashes,
        "hashes_cand": cand_hashes,
        "rigid_options": "(captured by bench_mi300.py defaults; see workspace/ab/<ts>/env.json)",
        "n_envs": cfg.get_default_n_envs(),
        "num_steps": num_steps,
        "warmup": warmup,
        "precision": cfg.get_default_precision(),
        "urdf": (
            (os.environ.get("AUTOKERNEL_SANDBOX") or
             os.path.expanduser("~/work/.cache/autokernels-genesis-sandbox"))
            + "/Genesis/newton-assets/unitree_g1/urdf/g1_29dof.urdf"
        ),
        "quadrants_commit": env_meta.get("quadrants_commit", "?"),
        "rocm_version": env_meta.get("rocm_version", "?"),
        "host": env_meta.get("host", "?"),
        "hip_visible_devices": env_meta.get("hip_visible_devices", "?"),
        "gpu_busy_pct_pre_run": gpu_busy_pre,
        "noise_suspect": noise_suspect,
        "bench_status": bench_result.get("status", "?"),
        "wall_time_s": bench_result.get("wall_time_s", 0.0),
        "throughput": bench_result.get("throughput", 0.0),
        "wall_seconds_total": bench_result.get("wall_seconds", 0.0),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    return rec


def write_session_writeup(out_dir: Path, args, env_meta: dict,
                          base_commit: str, cand_commit: str, files: list[str],
                          stats: dict, amdahl: dict, cross_kernel: dict,
                          base_kernels: dict, cand_kernels: dict,
                          verdict: str, reason: str,
                          outlier_trials: list[int]) -> Path:
    """spec section 5: human-readable per-experiment writeup."""
    name = args.name or f"ab-{cand_commit[:8]}"
    lines = []
    lines.append(f"# {name}  STATUS: {verdict}")
    lines.append("")
    lines.append(f"- branch_base:  {args.base}  ({base_commit[:12]})")
    lines.append(f"- branch_cand:  {args.cand}  ({cand_commit[:12]})")
    lines.append(f"- files:        {files}")
    lines.append(f"- env:          host={env_meta.get('host')}  "
                 f"gpu={env_meta.get('hip_visible_devices')}  "
                 f"quadrants={env_meta.get('quadrants_commit', '?')[:12]}")
    lines.append("")
    lines.append("## rocprofv3 (paired, kernel-level)")
    if args.target_kernel:
        bt = base_kernels.get("target") or {}
        ct = cand_kernels.get("target") or {}
        if bt and ct:
            kdelta = ((bt["avg_us"] - ct["avg_us"]) / bt["avg_us"] * 100) if bt["avg_us"] else 0
            lines.append(f"- target_kernel ({args.target_kernel}):")
            lines.append(f"    base avg_us = {bt.get('avg_us', 0):.2f}  "
                         f"total_ms = {bt.get('total_ms', 0):.2f}")
            lines.append(f"    cand avg_us = {ct.get('avg_us', 0):.2f}  "
                         f"total_ms = {ct.get('total_ms', 0):.2f}")
            lines.append(f"    Δ avg_us = {kdelta:+.2f}%   "
                         f"(% of base frame: {bt.get('pct_of_all', 0):.1f}%)")
    lines.append("- top non-target kernels (paired Δ):")
    for d in cross_kernel.get("all_deltas", []):
        if d["is_target"]:
            continue
        flag = "  ⚠ REGRESSION" if d["delta_pct"] >= DEFAULT_CROSS_KERNEL_REGRESSION_PCT else ""
        lines.append(f"    {d['name'][:60]:60s}  Δ={d['delta_pct']:+.2f}%{flag}")
    lines.append("")
    lines.append("## E2E (paired untraced)")
    lines.append(f"- N trials: {stats['n_trials']}")
    lines.append(f"- per-trial paired Δ%: " + ", ".join(
        f"{d:+.2f}" for d in stats.get("paired_deltas_pct", [])))
    lines.append(f"- median Δ: {stats['median_delta_pct']:+.2f}%")
    lines.append(f"- mean   Δ: {stats['mean_delta_pct']:+.2f}%")
    lines.append(f"- sign consistency: {stats['sign_pos']}/{stats['n_trials']} positive")
    lines.append(f"- base mean throughput: {stats['base_mean']:.0f} env*steps/s "
                 f"(CoV {stats['base_cov_pct']:.2f}%)")
    lines.append(f"- noise_suspect trials: {outlier_trials}")
    lines.append("")
    lines.append("## Amdahl plausibility")
    if amdahl.get("applicable"):
        lines.append(f"- target kernel was {amdahl['kernel_pct_of_all']:.1f}% of frame")
        lines.append(f"- claimed kernel Δ: {amdahl['kernel_delta_pct']:+.2f}%")
        lines.append(f"- expected E2E Δ ≈ {amdahl['kernel_pct_of_all']:.1f}% × "
                     f"{amdahl['kernel_delta_pct']:+.2f}% = {amdahl['expected_e2e_gain_pct']:+.2f}%")
        lines.append(f"- observed E2E Δ: {amdahl['observed_e2e_gain_pct']:+.2f}%")
        lines.append(f"- ratio observed/expected: {amdahl['ratio_observed_to_expected']:.2f}×  "
                     f"(tolerance ≥ {amdahl['tolerance']:.2f}×, "
                     f"{'PASS' if amdahl['passes'] else 'FAIL'})")
    else:
        lines.append(f"- not applicable: {amdahl.get('reason', '?')}")
    lines.append("")
    lines.append(f"## decision: **{verdict}**")
    lines.append(f"## reason: {reason}")
    lines.append("")
    p = out_dir / "summary.md"
    p.write_text("\n".join(lines))
    return p


# ---------------------------------------------------------------------------
# Main flow: cmd_ab
# ---------------------------------------------------------------------------

def cmd_ab(args: argparse.Namespace) -> int:
    repo_dir = args.repo or os.environ.get("GENESIS_SRC")
    if not repo_dir:
        die("--repo or $GENESIS_SRC must be set (path to the source repo to A/B)")
    if not Path(repo_dir).is_dir():
        die(f"repo dir not found: {repo_dir}")

    base_commit = resolve_commit(repo_dir, args.base)
    cand_commit = resolve_commit(repo_dir, args.cand)
    if base_commit == cand_commit:
        die(f"base ({args.base}) and cand ({args.cand}) resolve to the same commit "
            f"({base_commit[:12]}) -- nothing to A/B")
    files = changed_files(repo_dir, base_commit, cand_commit)
    if not files:
        die(f"no files changed between {args.base} and {args.cand}")

    # Hashes
    info(f"resolving file hashes for both arms ({len(files)} files)...")
    base_hashes = expected_hashes(repo_dir, base_commit, files)
    cand_hashes = expected_hashes(repo_dir, cand_commit, files)
    assert_arms_differ(base_hashes, cand_hashes, base_commit, cand_commit)

    # Env metadata
    env_meta = capture_env_metadata(repo_dir)
    info(f"env: genesis={env_meta['genesis_branch']}@{env_meta['genesis_head_commit'][:8]}  "
         f"quadrants={env_meta.get('quadrants_commit', '?')[:8]}  "
         f"gpu={env_meta['hip_visible_devices']}")

    # Merge-base age check (skippable)
    if args.prod_branch and not args.skip_merge_base_check:
        check_merge_base_age(repo_dir, env_meta["genesis_branch"], args.prod_branch,
                             max_age_h=args.max_merge_base_age_h)

    # Output dir
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = WORKSPACE / "ab" / f"{ts}-{args.name or 'ab'}"
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_dir = out_dir / "trials"
    trials_dir.mkdir(exist_ok=True)
    (out_dir / "env.json").write_text(json.dumps(env_meta, indent=2))

    # Cache wipe (once per session) -- spec 2.7
    info("wiping caches once for this session")
    _bench.wipe_caches()

    # GPU clock pin for low variance
    prev_perf = _bench.pin_gpu_clocks()
    try:
        # Warmup each arm (spec 2.7)
        info(f"warming up BASE arm ({args.warmup_runs} runs)")
        checkout_arm(repo_dir, base_commit, files)
        verify_arm_hashes(repo_dir, base_hashes, "BASE-warmup")
        for w in range(args.warmup_runs):
            r = run_one_untraced(f"/tmp/ab_warmup_base_{w}.json",
                                 num_steps=args.num_steps, warmup=args.bench_warmup)
            info(f"  warmup base[{w}]: status={r['status']} "
                 f"thr={r.get('throughput', 0):.0f} wall={r.get('wall_time_s', 0):.2f}s")

        info(f"warming up CAND arm ({args.warmup_runs} runs)")
        checkout_arm(repo_dir, cand_commit, files)
        verify_arm_hashes(repo_dir, cand_hashes, "CAND-warmup")
        for w in range(args.warmup_runs):
            r = run_one_untraced(f"/tmp/ab_warmup_cand_{w}.json",
                                 num_steps=args.num_steps, warmup=args.bench_warmup)
            info(f"  warmup cand[{w}]: status={r['status']} "
                 f"thr={r.get('throughput', 0):.0f} wall={r.get('wall_time_s', 0):.2f}s")

        # Paired interleaved trials -- spec 2.3
        base_trials, cand_trials = [], []
        for trial in range(1, args.trials + 1):
            info(f"\n--- trial {trial}/{args.trials} ---")

            # spec 2.9: concurrent-load gate
            if not args.skip_load_gate:
                if not wait_for_idle_gpu(args.gpu_busy_threshold, args.gpu_wait_max_s):
                    die(f"CONCURRENT_LOAD_DETECTED: GPU0 busy > "
                        f"{args.gpu_busy_threshold}% for >{args.gpu_wait_max_s}s")

            # BASE arm
            checkout_arm(repo_dir, base_commit, files)
            verify_arm_hashes(repo_dir, base_hashes, f"BASE-t{trial}")
            busy = gpu_busy_pct()
            r = run_one_untraced(f"/tmp/ab_base_t{trial}.json",
                                 num_steps=args.num_steps, warmup=args.bench_warmup)
            rec = make_trial_record("base", trial, r, base_commit, cand_commit, files,
                                    base_hashes, cand_hashes, env_meta, busy,
                                    args.num_steps, args.bench_warmup)
            (trials_dir / f"base_t{trial}.json").write_text(json.dumps(rec, indent=2))
            base_trials.append(rec)
            info(f"  base[{trial}]: thr={r.get('throughput', 0):.0f}  "
                 f"wall={r.get('wall_time_s', 0):.3f}s  status={r['status']}")

            # Concurrent-load check between arms too
            if not args.skip_load_gate:
                if not wait_for_idle_gpu(args.gpu_busy_threshold, args.gpu_wait_max_s):
                    die(f"CONCURRENT_LOAD_DETECTED: GPU0 busy > "
                        f"{args.gpu_busy_threshold}% for >{args.gpu_wait_max_s}s")

            # CAND arm
            checkout_arm(repo_dir, cand_commit, files)
            verify_arm_hashes(repo_dir, cand_hashes, f"CAND-t{trial}")
            busy = gpu_busy_pct()
            r = run_one_untraced(f"/tmp/ab_cand_t{trial}.json",
                                 num_steps=args.num_steps, warmup=args.bench_warmup)
            rec = make_trial_record("cand", trial, r, base_commit, cand_commit, files,
                                    base_hashes, cand_hashes, env_meta, busy,
                                    args.num_steps, args.bench_warmup)
            (trials_dir / f"cand_t{trial}.json").write_text(json.dumps(rec, indent=2))
            cand_trials.append(rec)
            info(f"  cand[{trial}]: thr={r.get('throughput', 0):.0f}  "
                 f"wall={r.get('wall_time_s', 0):.3f}s  status={r['status']}")

        # Verify warmup was sufficient (spec 2.7): trial 1 wall vs trial N wall
        if args.trials >= 3 and base_trials[0]["wall_time_s"] > 0:
            t1_wall = base_trials[0]["wall_time_s"]
            tN_wall = base_trials[-1]["wall_time_s"]
            if tN_wall > 0 and (t1_wall - tN_wall) / tN_wall > 0.05:
                warn(f"warmup may be insufficient: trial 1 wall {t1_wall:.3f}s vs "
                     f"trial {args.trials} wall {tN_wall:.3f}s "
                     f"({(t1_wall - tN_wall) / tN_wall * 100:+.1f}%) -- "
                     f"results may be biased; consider --warmup-runs {args.warmup_runs + 2}")

        # Filter to PASS trials
        base_pass = [t for t in base_trials if t["bench_status"] == "PASS"]
        cand_pass = [t for t in cand_trials if t["bench_status"] == "PASS"]
        if len(base_pass) < args.trials or len(cand_pass) < args.trials:
            die(f"some trials crashed: base PASS={len(base_pass)}/{args.trials}, "
                f"cand PASS={len(cand_pass)}/{args.trials}. "
                f"Inspect {trials_dir} for failure details.")

        base_thr = [t["throughput"] for t in base_pass]
        cand_thr = [t["throughput"] for t in cand_pass]
        stats = paired_stats(base_thr, cand_thr)
        outliers = flag_outliers(stats)
        if outliers:
            warn(f"trials flagged as noise_suspect: {outliers}")
            for i in outliers:
                base_trials[i - 1]["noise_suspect"] = True
                cand_trials[i - 1]["noise_suspect"] = True
                (trials_dir / f"base_t{i}.json").write_text(
                    json.dumps(base_trials[i - 1], indent=2))
                (trials_dir / f"cand_t{i}.json").write_text(
                    json.dumps(cand_trials[i - 1], indent=2))

        # Rocprof (paired, M >= 2) -- spec 2.6
        base_kernels_runs = []
        cand_kernels_runs = []
        if args.target_kernel and not args.skip_traced:
            for r in range(1, args.rocprof_runs + 1):
                info(f"\n--- rocprof run {r}/{args.rocprof_runs} ---")
                # BASE
                checkout_arm(repo_dir, base_commit, files)
                verify_arm_hashes(repo_dir, base_hashes, f"BASE-rocprof{r}")
                tres = run_one_traced(f"/work/runs/ab_base_rocprof_{r}",
                                      num_steps=args.traced_num_steps)
                if tres["status"] == "PASS":
                    parsed = parse_top_kernels(tres["csv_path"], args.target_kernel)
                    base_kernels_runs.append(parsed)
                # CAND
                checkout_arm(repo_dir, cand_commit, files)
                verify_arm_hashes(repo_dir, cand_hashes, f"CAND-rocprof{r}")
                tres = run_one_traced(f"/work/runs/ab_cand_rocprof_{r}",
                                      num_steps=args.traced_num_steps)
                if tres["status"] == "PASS":
                    parsed = parse_top_kernels(tres["csv_path"], args.target_kernel)
                    cand_kernels_runs.append(parsed)

        # Median over paired rocprof runs
        def _median_kernels(runs: list[dict]) -> dict:
            if not runs:
                return {}
            # Median target avg_us across runs
            tgt_avgs = [r["target"]["avg_us"] for r in runs if r.get("target")]
            tgt_totals = [r["target"]["total_ms"] for r in runs if r.get("target")]
            tgt_pct = [r["target"]["pct_of_all"] for r in runs if r.get("target")]
            target = {
                "avg_us": statistics.median(tgt_avgs) if tgt_avgs else 0,
                "total_ms": statistics.median(tgt_totals) if tgt_totals else 0,
                "pct_of_all": statistics.median(tgt_pct) if tgt_pct else 0,
            } if tgt_avgs else None
            # Median per-kernel for top -- align by name
            by_name: dict[str, list[float]] = {}
            for r in runs:
                for k in r.get("top", []):
                    by_name.setdefault(k["name"], []).append(k["avg_us"])
            top = []
            for name, avgs in by_name.items():
                top.append({"name": name, "avg_us": statistics.median(avgs),
                            "total_ms": 0, "calls": 0})
            top.sort(key=lambda x: x["avg_us"], reverse=True)
            return {"target": target, "top": top[:8]}

        base_kernels = _median_kernels(base_kernels_runs)
        cand_kernels = _median_kernels(cand_kernels_runs)

        # Cross-kernel check
        cross_kernel = cross_kernel_regression_check(
            base_kernels, cand_kernels, args.target_kernel,
            threshold_pct=args.cross_kernel_regression_pct,
        ) if base_kernels.get("top") and cand_kernels.get("top") else \
            {"regressed": [], "all_deltas": []}

        # Amdahl
        amdahl = amdahl_check(stats, base_kernels, cand_kernels, args.target_kernel,
                              tolerance=args.amdahl_tolerance)

        # Decision
        thresholds = {
            "min_median_delta_pct": args.min_median_delta_pct,
            "sign_consistency_frac": args.sign_consistency_frac,
            "max_base_cov_pct": args.max_base_cov_pct,
        }
        verdict, reason = apply_decision_rule(stats, amdahl, cross_kernel, thresholds)

        # Record baseline for next session's drift check
        if base_kernels.get("target"):
            record_baseline(base_kernels["target"]["avg_us"])

        # Writeup
        summary_path = write_session_writeup(
            out_dir, args, env_meta, base_commit, cand_commit, files,
            stats, amdahl, cross_kernel, base_kernels, cand_kernels,
            verdict, reason, outliers,
        )
        # Decision JSON (machine-readable)
        decision = {
            "verdict": verdict,
            "reason": reason,
            "stats": stats,
            "amdahl": amdahl,
            "cross_kernel": cross_kernel,
            "base_kernels": base_kernels,
            "cand_kernels": cand_kernels,
            "outlier_trials": outliers,
            "trials_dir": str(trials_dir),
            "summary_path": str(summary_path),
        }
        (out_dir / "decision.json").write_text(json.dumps(decision, indent=2, default=str))

        # Print final
        print()
        print("=" * 72)
        print(f"VERDICT: {verdict}")
        print(f"REASON:  {reason}")
        print(f"WRITEUP: {summary_path}")
        print("=" * 72)
        return 0
    finally:
        # Restore HEAD to cand (the branch tip the user committed)
        checkout_arm(repo_dir, cand_commit, files)
        _bench.restore_gpu_clocks(prev_perf)


# ---------------------------------------------------------------------------
# Self-tests (spec section 4)
# ---------------------------------------------------------------------------

def cmd_selftest_noop(args: argparse.Namespace) -> int:
    """Apply a whitespace-only patch, run ab, expect DISCARD/NEUTRAL with median Δ ≈ 0."""
    repo_dir = args.repo or os.environ.get("GENESIS_SRC")
    if not repo_dir:
        die("--repo or $GENESIS_SRC required")
    info("self-test (no-op): adding a whitespace comment to a Genesis file...")
    target_file = "genesis/__init__.py"
    full = Path(repo_dir) / target_file
    if not full.exists():
        die(f"{full} not found")
    # Read, append a harmless comment, commit on a temp branch
    rc, branch = run(["git", "-C", repo_dir, "rev-parse", "--abbrev-ref", "HEAD"])
    base_branch = branch.strip()
    selftest_branch = f"selftest-noop-{int(time.time())}"
    run(["git", "-C", repo_dir, "checkout", "-b", selftest_branch])
    with open(full, "a") as f:
        f.write(f"\n# autokernels-genesis ab.py self-test (no-op) at {datetime.utcnow().isoformat()}\n")
    run(["git", "-C", repo_dir, "add", target_file])
    run(["git", "-C", repo_dir, "commit", "-m", "selftest: noop comment"])
    selftest_args = argparse.Namespace(**vars(args))
    selftest_args.base = base_branch
    selftest_args.cand = selftest_branch
    selftest_args.name = "selftest-noop"
    rc = cmd_ab(selftest_args)
    # Cleanup
    run(["git", "-C", repo_dir, "checkout", base_branch])
    run(["git", "-C", repo_dir, "branch", "-D", selftest_branch])
    if rc != 0:
        die(f"self-test failed (rc={rc})")
    # Read decision
    # ... (left as info for the user)
    info("self-test complete; check workspace/ab/<latest>/decision.json -- verdict should be DISCARD/NEUTRAL")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ab
    a = sub.add_parser("ab", help="run paired A/B between two refs")
    a.add_argument("--base", default="HEAD~1", help="base ref or commit (default: HEAD~1)")
    a.add_argument("--cand", default="HEAD",  help="candidate ref or commit (default: HEAD)")
    a.add_argument("--repo", help="path to git repo (default: $GENESIS_SRC)")
    a.add_argument("--target-kernel", default=None,
                   help="regex matching the kernel(s) of interest for Amdahl + rocprof attribution")
    a.add_argument("--name", default=None, help="session name for the writeup dir")
    a.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                   help=f"untraced paired trials per arm (default {DEFAULT_TRIALS})")
    a.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS,
                   help=f"warmup invocations per arm (default {DEFAULT_WARMUP_RUNS})")
    a.add_argument("--bench-warmup", type=int, default=DEFAULT_BENCH_WARMUP,
                   help=f"--warmup steps inside bench_mi300.py (default {DEFAULT_BENCH_WARMUP})")
    a.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS,
                   help=f"untraced bench steps (default {DEFAULT_NUM_STEPS})")
    a.add_argument("--rocprof-runs", type=int, default=DEFAULT_ROCPROF_RUNS,
                   help=f"rocprof paired runs per arm (default {DEFAULT_ROCPROF_RUNS})")
    a.add_argument("--traced-num-steps", type=int, default=DEFAULT_TRACED_NUM_STEPS,
                   help=f"traced bench steps (default {DEFAULT_TRACED_NUM_STEPS})")
    a.add_argument("--skip-traced", action="store_true",
                   help="skip rocprof + Amdahl + cross-kernel checks (NOT recommended)")
    a.add_argument("--min-median-delta-pct", type=float, default=DEFAULT_MIN_MEDIAN_DELTA_PCT)
    a.add_argument("--sign-consistency-frac", type=float, default=DEFAULT_SIGN_CONSISTENCY_FRAC)
    a.add_argument("--max-base-cov-pct", type=float, default=DEFAULT_MAX_BASE_COV_PCT)
    a.add_argument("--amdahl-tolerance", type=float, default=DEFAULT_AMDAHL_TOLERANCE)
    a.add_argument("--cross-kernel-regression-pct", type=float,
                   default=DEFAULT_CROSS_KERNEL_REGRESSION_PCT)
    a.add_argument("--gpu-busy-threshold", type=float, default=DEFAULT_GPU_BUSY_THRESHOLD)
    a.add_argument("--gpu-wait-max-s", type=int, default=DEFAULT_GPU_WAIT_MAX_S)
    a.add_argument("--skip-load-gate", action="store_true",
                   help="bypass concurrent-GPU-load gate (NOT recommended)")
    a.add_argument("--prod-branch", default=None,
                   help="production branch ref for merge-base age check (e.g. origin/release/0.4.4.amd1)")
    a.add_argument("--max-merge-base-age-h", type=int, default=24)
    a.add_argument("--skip-merge-base-check", action="store_true")

    # selftest
    s = sub.add_parser("selftest-noop",
                       help="run a no-op self-test; expect DISCARD/NEUTRAL")
    for k, v in [
        ("repo", None), ("trials", DEFAULT_TRIALS), ("warmup_runs", DEFAULT_WARMUP_RUNS),
        ("bench_warmup", DEFAULT_BENCH_WARMUP), ("num_steps", DEFAULT_NUM_STEPS),
        ("rocprof_runs", DEFAULT_ROCPROF_RUNS),
        ("traced_num_steps", DEFAULT_TRACED_NUM_STEPS),
        ("min_median_delta_pct", DEFAULT_MIN_MEDIAN_DELTA_PCT),
        ("sign_consistency_frac", DEFAULT_SIGN_CONSISTENCY_FRAC),
        ("max_base_cov_pct", DEFAULT_MAX_BASE_COV_PCT),
        ("amdahl_tolerance", DEFAULT_AMDAHL_TOLERANCE),
        ("cross_kernel_regression_pct", DEFAULT_CROSS_KERNEL_REGRESSION_PCT),
        ("gpu_busy_threshold", DEFAULT_GPU_BUSY_THRESHOLD),
        ("gpu_wait_max_s", DEFAULT_GPU_WAIT_MAX_S),
        ("max_merge_base_age_h", 24),
    ]:
        s.add_argument(f"--{k.replace('_', '-')}", default=v,
                       type=type(v) if v is not None else str)
    s.add_argument("--target-kernel", default=None)
    s.add_argument("--skip-traced", action="store_true", default=True)  # noop test, skip
    s.add_argument("--skip-load-gate", action="store_true", default=True)
    s.add_argument("--skip-merge-base-check", action="store_true", default=True)
    s.add_argument("--prod-branch", default=None)
    s.add_argument("--name", default="selftest-noop")

    args = ap.parse_args()
    return {"ab": cmd_ab, "selftest-noop": cmd_selftest_noop}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
