#!/usr/bin/env python3
"""
watchdog.py -- safety daemon for indefinite autokernels-genesis runs.

The agent loop never stops on its own (orchestrate.py only returns CONTINUE/HALT).
This daemon is the *only* component allowed to set workspace/HALT.flag, and it
does so for narrow, recoverable infrastructure reasons (not for scientific
plateaus). It also handles routine housekeeping that an indefinite loop needs:

  - log rotation: move run.log -> workspace/logs/expN.log per experiment
  - disk: warn at 85%, HALT at 95% of /home (or whatever holds GENESIS_SRC)
  - git: weekly `git gc --auto` per worktree
  - drift detection: every N experiments, re-run an unedited-baseline bench
                     and HALT if throughput drifts > 2 * recorded sigma

Usage:
  uv run watchdog.py                     # one-shot check + housekeeping
  uv run watchdog.py --loop --interval 300   # daemon mode, every 5 min
  uv run watchdog.py --clear-halt        # remove HALT.flag (after fix)
  uv run watchdog.py --rotate-logs       # rotate run.log -> workspace/logs/

Designed to run in a SEPARATE process from the agent (cron, systemd, or a
plain `nohup uv run watchdog.py --loop &` next to the agent loop). The agent
checks for HALT.flag at the top of every iteration via orchestrate.py next.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR / "workspace"
LOGS_DIR = WORKSPACE / "logs"
HALT_FLAG = WORKSPACE / "HALT.flag"
RESULTS_TSV = SCRIPT_DIR / "results.tsv"
RUN_LOG = SCRIPT_DIR / "run.log"
STATE_PATH = WORKSPACE / "orchestration_state.json"
DRIFT_STATE = WORKSPACE / "watchdog_drift.json"

DISK_WARN_PCT = 85
DISK_HALT_PCT = 95
LOG_ROTATE_KEEP = 200
GIT_GC_INTERVAL_S = 7 * 24 * 3600  # weekly
DRIFT_CHECK_EVERY_N = 50           # experiments (single-stream check)
DRIFT_GLOBAL_WINDOW = 10           # KEEPs per window for cross-agent slope test
DRIFT_GLOBAL_MIN_GPUS = 3          # need >=3 active GPUs to call infra drift
DRIFT_GLOBAL_SIGMA_K = 3.0         # K * baseline_sigma_pct triggers HALT
DRIFT_GLOBAL_MIN_TOTAL = 20        # need >=20 KEEP rows to even attempt the test


def log(msg: str) -> None:
    print(f"[watchdog {datetime.utcnow().isoformat()}Z] {msg}", flush=True)


def set_halt(reason: str) -> None:
    WORKSPACE.mkdir(exist_ok=True)
    HALT_FLAG.write_text(f"{datetime.utcnow().isoformat()}Z {reason}\n")
    log(f"HALT set: {reason}")


def clear_halt() -> None:
    if HALT_FLAG.exists():
        HALT_FLAG.unlink()
        log("HALT cleared")


# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------

def check_disk() -> None:
    target = SCRIPT_DIR
    usage = shutil.disk_usage(target)
    pct = 100 * usage.used / usage.total
    if pct >= DISK_HALT_PCT:
        set_halt(f"disk at {pct:.1f}% on {target} (>= {DISK_HALT_PCT}%)")
    elif pct >= DISK_WARN_PCT:
        log(f"WARN disk at {pct:.1f}% on {target}")


# ---------------------------------------------------------------------------
# Log rotation
# ---------------------------------------------------------------------------

def _last_experiment_n() -> int:
    if not RESULTS_TSV.exists():
        return 0
    try:
        with open(RESULTS_TSV) as f:
            rows = f.read().strip().splitlines()
        if len(rows) <= 1:
            return 0
        last = rows[-1].split("\t")
        return int(last[0])
    except (OSError, ValueError, IndexError):
        return 0


def rotate_logs() -> None:
    if not RUN_LOG.exists() or RUN_LOG.stat().st_size == 0:
        return
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    n = _last_experiment_n()
    dst = LOGS_DIR / f"exp{n}.log"
    # Don't clobber if the agent re-ran without recording
    suffix = 0
    while dst.exists():
        suffix += 1
        dst = LOGS_DIR / f"exp{n}-r{suffix}.log"
    shutil.copy2(RUN_LOG, dst)
    RUN_LOG.write_text("")  # truncate, don't unlink (agent may have it open)
    log(f"rotated run.log -> {dst.name}")
    _prune_logs()


def _prune_logs() -> None:
    if not LOGS_DIR.exists():
        return
    files = sorted(LOGS_DIR.glob("exp*.log"), key=lambda p: p.stat().st_mtime)
    excess = len(files) - LOG_ROTATE_KEEP
    if excess > 0:
        for p in files[:excess]:
            try:
                p.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Git GC
# ---------------------------------------------------------------------------

def _last_gc_path() -> Path:
    return WORKSPACE / "watchdog_last_gc"


def maybe_git_gc() -> None:
    last_gc = _last_gc_path()
    now = time.time()
    if last_gc.exists():
        try:
            ts = float(last_gc.read_text().strip())
            if (now - ts) < GIT_GC_INTERVAL_S:
                return
        except (OSError, ValueError):
            pass

    targets = []
    genesis_src = os.environ.get("GENESIS_SRC")
    if genesis_src and os.path.isdir(os.path.join(genesis_src, ".git")):
        targets.append(genesis_src)
    targets.append(str(SCRIPT_DIR))

    for t in targets:
        try:
            subprocess.run(["git", "-C", t, "gc", "--auto"],
                           capture_output=True, timeout=120)
            log(f"git gc --auto on {t}")
        except (subprocess.SubprocessError, OSError) as e:
            log(f"git gc failed on {t}: {e}")

    WORKSPACE.mkdir(exist_ok=True)
    last_gc.write_text(str(now))


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def _load_drift_state() -> dict:
    if not DRIFT_STATE.exists():
        return {"last_check_exp": 0, "history": []}
    try:
        return json.loads(DRIFT_STATE.read_text())
    except (OSError, ValueError):
        return {"last_check_exp": 0, "history": []}


def _save_drift_state(s: dict) -> None:
    WORKSPACE.mkdir(exist_ok=True)
    DRIFT_STATE.write_text(json.dumps(s, indent=2))


def _baseline_calibration() -> dict | None:
    p = WORKSPACE / "baseline_calibration.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def maybe_drift_check(campaign: str | None) -> None:
    """Every N experiments, re-bench the unedited baseline commit and compare.

    'Unedited baseline' = the merge-base of the campaign branch with main, not
    the current HEAD (which has all the agent's kept commits). This requires
    git stash + checkout + bench + restore, which is intrusive; we skip if
    GENESIS_SRC is dirty or if no calibration exists.
    """
    if not campaign:
        return
    cal = _baseline_calibration()
    if not cal or "median_throughput" not in cal:
        return

    n = _last_experiment_n()
    state = _load_drift_state()
    if (n - state.get("last_check_exp", 0)) < DRIFT_CHECK_EVERY_N:
        return

    log(f"drift check triggered at exp{n} (last check exp{state.get('last_check_exp', 0)})")
    # Heuristic: just consult the most recent baseline-tagged row in results.tsv.
    # A fully invasive re-bench requires coordinating with the agent (it might
    # be mid-edit). Conservative: check that the rolling median of the last 5
    # baseline-tagged or recently-KEPT rows is within 2*sigma of the original.
    baseline_thr = cal["median_throughput"]
    baseline_sigma_pct = cal.get("e2e_noise_floor_pct", 1.0)
    threshold_low = baseline_thr * (1 - 2 * baseline_sigma_pct / 100)
    threshold_high = baseline_thr * (1 + 5 * baseline_sigma_pct / 100)

    recent = _recent_throughputs(5)
    if recent:
        median = sorted(recent)[len(recent) // 2]
        if median < threshold_low:
            set_halt(f"drift: rolling median {median:.0f} < threshold {threshold_low:.0f} "
                     f"(baseline {baseline_thr:.0f} +/- {baseline_sigma_pct:.1f}%)")
        else:
            log(f"drift OK: median {median:.0f} within [{threshold_low:.0f}, {threshold_high:.0f}]")

    state["last_check_exp"] = n
    state.setdefault("history", []).append({
        "ts": datetime.utcnow().isoformat() + "Z",
        "exp": n,
        "median_recent_5": (sorted(recent)[len(recent) // 2] if recent else None),
        "baseline": baseline_thr,
    })
    _save_drift_state(state)


def _recent_throughputs(n: int) -> list[float]:
    if not RESULTS_TSV.exists():
        return []
    out: list[float] = []
    try:
        with open(RESULTS_TSV) as f:
            rows = f.read().strip().splitlines()
        for line in rows[-n - 1:]:  # +1 to allow header skip
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            try:
                out.append(float(parts[5]))
            except ValueError:
                continue
    except OSError:
        return []
    return out


# ---------------------------------------------------------------------------
# Cross-agent statistical drift detection
# ---------------------------------------------------------------------------
# Reads the shared global_log.tsv (populated by all 8 agents) and tests:
#   "Has the cross-agent KEEP-throughput median trended down beyond what
#    baseline noise can explain?"
# This is the closest defensible analog to AutoResearch's deterministic eval:
# we can't re-bench the unedited baseline without coordinating with agents
# (which would violate NEVER STOP), but agents only KEEP improvements -- so
# *later* keeps should be >= *earlier* keeps. If they're significantly
# worse, the floor under everyone is moving and that's infra, not science.

def _shared_global_log() -> Path | None:
    explicit = os.environ.get("AUTOKERNEL_SHARED_DIR")
    if explicit:
        p = Path(explicit) / "global_log.tsv"
    else:
        workspace_root = os.environ.get("AUTOKERNEL_ROOT") or os.path.expanduser("~/work")
        p = Path(workspace_root) / "autokernels-shared" / "global_log.tsv"
    return p if p.exists() else None


def _load_global_keeps(campaign: str | None) -> list[dict]:
    """Read KEEP rows from the shared global log. Tolerant of partial last lines."""
    p = _shared_global_log()
    if not p:
        return []
    import csv as _csv
    out: list[dict] = []
    try:
        text = p.read_text()
    except OSError:
        return []
    for r in _csv.DictReader(text.splitlines(), delimiter="\t"):
        if (r.get("status") or "").lower() != "keep":
            continue
        if campaign and r.get("campaign") != campaign:
            continue
        try:
            r["_thr"] = float(r.get("e2e_throughput") or 0.0)
        except (TypeError, ValueError):
            continue
        if r["_thr"] <= 0:
            continue
        out.append(r)
    return out


def maybe_drift_check_global(campaign: str | None) -> None:
    """Cross-agent statistical drift check.

    Splits the recent KEEP rows into two equal windows (earlier vs. recent) of
    DRIFT_GLOBAL_WINDOW each. Since agents only KEEP improvements, later-window
    median should be >= earlier-window median by the noise floor. If recent
    median has dropped by more than DRIFT_GLOBAL_SIGMA_K * baseline_sigma_pct,
    declare infra drift and HALT.

    Guards against false positives:
      - require >=DRIFT_GLOBAL_MIN_TOTAL total KEEPs
      - require >=DRIFT_GLOBAL_MIN_GPUS distinct GPUs in the recent window
        (single bad GPU shouldn't HALT the fleet)
    """
    cal = _baseline_calibration()
    if not cal:
        return
    sigma_pct = float(cal.get("e2e_noise_floor_pct", 1.0))

    keeps = _load_global_keeps(campaign)
    win = DRIFT_GLOBAL_WINDOW
    if len(keeps) < max(DRIFT_GLOBAL_MIN_TOTAL, 2 * win):
        return

    recent = keeps[-win:]
    earlier = keeps[-2 * win:-win]
    recent_gpus = {r.get("gpu_id") for r in recent}
    if len(recent_gpus) < DRIFT_GLOBAL_MIN_GPUS:
        return  # not enough cross-GPU coverage to call it infra

    def _median(rows: list[dict]) -> float:
        xs = sorted(r["_thr"] for r in rows)
        return xs[len(xs) // 2]

    med_recent = _median(recent)
    med_earlier = _median(earlier)
    delta_pct = 100.0 * (med_recent - med_earlier) / med_earlier
    threshold_pct = -DRIFT_GLOBAL_SIGMA_K * sigma_pct

    log(f"global drift: recent_median={med_recent:.0f} earlier_median={med_earlier:.0f} "
        f"delta={delta_pct:+.2f}%  threshold<{threshold_pct:+.2f}%  "
        f"gpus={sorted(recent_gpus)}")

    if delta_pct < threshold_pct:
        set_halt(
            f"cross-agent drift: KEEP-throughput median dropped {delta_pct:+.2f}% "
            f"(< {threshold_pct:+.2f}% threshold from baseline sigma {sigma_pct:.2f}%); "
            f"recent median {med_recent:.0f} vs earlier {med_earlier:.0f} "
            f"across gpus {sorted(recent_gpus)}. Re-run baseline outside the loop "
            f"and clear HALT once verified."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def one_pass(campaign: str | None) -> None:
    check_disk()
    rotate_logs()
    maybe_git_gc()
    maybe_drift_check(campaign)
    maybe_drift_check_global(campaign)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true", help="run as daemon")
    ap.add_argument("--interval", type=int, default=300, help="seconds between passes (with --loop)")
    ap.add_argument("--campaign", default=os.environ.get("CAMPAIGN"),
                    help="campaign id for drift check (default: $CAMPAIGN)")
    ap.add_argument("--clear-halt", action="store_true", help="remove workspace/HALT.flag and exit")
    ap.add_argument("--rotate-logs", action="store_true", help="just rotate run.log and exit")
    args = ap.parse_args()

    if args.clear_halt:
        clear_halt()
        return 0
    if args.rotate_logs:
        rotate_logs()
        return 0

    if args.loop:
        log(f"daemon mode, interval={args.interval}s, campaign={args.campaign}")
        try:
            while True:
                try:
                    one_pass(args.campaign)
                except Exception as e:
                    log(f"pass failed: {e!r}")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            log("interrupted, exiting")
        return 0
    else:
        one_pass(args.campaign)
        return 0


if __name__ == "__main__":
    sys.exit(main())
