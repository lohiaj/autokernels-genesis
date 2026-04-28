#!/usr/bin/env python3
"""
orchestrate.py -- track campaign state + emit advisory signals.

Reads/writes workspace/orchestration_state.json (initialized by prepare.py).

Subcommands:
  uv run orchestrate.py status                                              -- print campaign state
  uv run orchestrate.py next --campaign <id>                                -- always CONTINUE (+stderr advisories)
  uv run orchestrate.py recommend --campaign <id>                          -- print structured recommendations
  uv run orchestrate.py record --campaign <id> --kernel-avg-us X --e2e-throughput Y \
                               --status keep|revert|crash --description "..."
  uv run orchestrate.py report                                              -- write workspace/report.md
  uv run orchestrate.py reset                                               -- nuke state (use with care)

NOTE: As of the loop redesign, `next` NEVER returns DONE. The only legitimate
stop is human SIGINT or the watchdog setting workspace/HALT.flag.
The previous move-on criteria are now soft advisories printed to stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR / "workspace"
STATE_PATH = WORKSPACE / "orchestration_state.json"
H100_REF = 794280.0


def load_state() -> dict:
    if not STATE_PATH.exists():
        die(f"{STATE_PATH} missing. Run prepare.py first.")
    with open(STATE_PATH) as f:
        return json.load(f)


def save_state(state: dict) -> None:
    state["updated_at"] = datetime.utcnow().isoformat() + "Z"
    WORKSPACE.mkdir(exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def get_campaign(state: dict, name: str) -> dict:
    for c in state["campaigns"]:
        if c["campaign_id"] == name:
            return c
    die(f"campaign '{name}' not in state. known: {[c['campaign_id'] for c in state['campaigns']]}")


def die(msg: str) -> None:
    print(f"orchestrate: ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_status(state: dict) -> None:
    print(f"orchestrator state ({STATE_PATH})")
    print(f"  noise_floor_pct: {state.get('noise_floor_pct')}")
    print(f"  baseline_e2e_throughput: {state.get('baseline_e2e_throughput')}")
    print(f"  H100 reference: {H100_REF:.0f} env*steps/s")
    print()
    print("campaigns:")
    for c in state["campaigns"]:
        cur_pct = c.get("current_pct_of_h100", 0)
        target = c.get("target_pct_of_h100", 80)
        baseline_us = c.get("baseline_kernel_avg_us")
        best_us = c.get("best_kernel_avg_us")
        best_thr = c.get("best_e2e_throughput")
        runs = c.get("experiments_run", 0)
        reverts = c.get("consecutive_reverts", 0)
        status = c.get("status", "pending")
        kernel_delta = ""
        if baseline_us and best_us:
            d = 100 * (1 - best_us / baseline_us)
            kernel_delta = f"  kernel: {baseline_us:.1f}us -> {best_us:.1f}us ({d:+.1f}%)"
        thr_str = f"  best_e2e: {best_thr:.0f} env*steps/s" if best_thr else ""
        print(f"  [{status:11s}] {c['campaign_id']}: {cur_pct:.1f}% -> target {target:.0f}%{kernel_delta}{thr_str}")
        print(f"               experiments={runs}  consecutive_reverts={reverts}")


def _campaign_advisories(c: dict, crit: dict) -> list[str]:
    """Compute non-blocking warnings for a campaign. Never causes a stop."""
    out: list[str] = []
    rev_warn = crit.get("warn_consecutive_reverts", 5)
    rev_high = crit.get("consecutive_reverts", 8)
    if c.get("consecutive_reverts", 0) >= rev_high:
        out.append(
            f"WARN consecutive_reverts={c['consecutive_reverts']} (>= {rev_high}). "
            "Re-read workspace/learning.md, switch hypothesis class, or run --profile-omniperf."
        )
    elif c.get("consecutive_reverts", 0) >= rev_warn:
        out.append(
            f"NOTE consecutive_reverts={c['consecutive_reverts']}. Consider varying idea class."
        )
    if c.get("experiments_run", 0) >= crit.get("max_experiments_per_campaign", 80):
        out.append(
            f"NOTE experiments_run={c['experiments_run']}. Soft budget reached; keep going if EV remains positive."
        )
    if c.get("current_pct_of_h100", 0) >= crit.get("kernel_pct_target", 80.0):
        out.append(
            f"NOTE current_pct_of_h100={c['current_pct_of_h100']:.1f}% >= target. Additional gains compound."
        )
    return out


def _halt_flag_path() -> Path:
    return WORKSPACE / "HALT.flag"


def cmd_next(state: dict, campaign_name: str) -> None:
    """Always CONTINUE unless the watchdog has set HALT.flag.

    Advisories print to stderr so the agent sees them but the contract on stdout
    stays simple: 'CONTINUE' or 'HALT'.
    """
    c = get_campaign(state, campaign_name)
    crit = state.get("move_on_criteria", {})

    # Reflect optimizing status if not already
    if c.get("status") in (None, "pending"):
        c["status"] = "optimizing"
        save_state(state)

    for line in _campaign_advisories(c, crit):
        print(line, file=sys.stderr)

    halt = _halt_flag_path()
    if halt.exists():
        try:
            reason = halt.read_text().strip() or "watchdog set HALT.flag"
        except OSError:
            reason = "watchdog set HALT.flag (unreadable)"
        print(f"HALT  reason={reason}", file=sys.stderr)
        print("HALT")
        return

    print("CONTINUE")


def cmd_recommend(state: dict, campaign_name: str) -> None:
    """Structured recommendation for the agent (advisory only)."""
    c = get_campaign(state, campaign_name)
    crit = state.get("move_on_criteria", {})

    print(f"campaign={campaign_name}")
    print(f"experiments_run={c.get('experiments_run', 0)}")
    print(f"consecutive_reverts={c.get('consecutive_reverts', 0)}")
    print(f"current_pct_of_h100={c.get('current_pct_of_h100', 0):.2f}")
    print(f"best_kernel_avg_us={c.get('best_kernel_avg_us')}")
    print(f"best_e2e_throughput={c.get('best_e2e_throughput')}")
    print()
    advisories = _campaign_advisories(c, crit)
    if not advisories:
        print("status: healthy -- continue with playbook + ideas.md")
    else:
        print("advisories:")
        for line in advisories:
            print(f"  - {line}")


def cmd_record(state: dict, *, campaign_name: str, kernel_avg_us: float,
               e2e_throughput: float, status: str, description: str) -> None:
    c = get_campaign(state, campaign_name)
    c["experiments_run"] = c.get("experiments_run", 0) + 1
    c["last_description"] = description
    c["last_kernel_avg_us"] = kernel_avg_us
    c["last_e2e_throughput"] = e2e_throughput
    c["last_status"] = status

    if status == "keep":
        c["consecutive_reverts"] = 0
        # Update best if this is better
        best_us = c.get("best_kernel_avg_us")
        if best_us is None or (kernel_avg_us > 0 and kernel_avg_us < best_us):
            c["best_kernel_avg_us"] = kernel_avg_us
        best_thr = c.get("best_e2e_throughput")
        if best_thr is None or (e2e_throughput > best_thr):
            c["best_e2e_throughput"] = e2e_throughput
            c["current_pct_of_h100"] = 100.0 * e2e_throughput / H100_REF
        c["current_kernel_avg_us"] = kernel_avg_us
        c["status"] = "optimizing"
    else:
        c["consecutive_reverts"] = c.get("consecutive_reverts", 0) + 1

    save_state(state)
    runs = c["experiments_run"]
    reverts = c["consecutive_reverts"]
    print(f"recorded: experiment={runs}  consecutive_reverts={reverts}  status={status}")


def cmd_report(state: dict) -> None:
    out = WORKSPACE / "report.md"
    lines = []
    lines.append("# autokernels-genesis report")
    lines.append("")
    lines.append(f"_generated {datetime.utcnow().isoformat()}Z_")
    lines.append("")
    lines.append(f"H100 reference: {H100_REF:.0f} env*steps/s")
    lines.append(f"Baseline e2e (pre-run): {state.get('baseline_e2e_throughput')}")
    lines.append("")
    lines.append("## Campaign results")
    lines.append("")
    lines.append("| Campaign | Status | Experiments | Best kernel_avg_us | Best e2e | % of H100 |")
    lines.append("|---|---|---|---|---|---|")
    for c in state["campaigns"]:
        lines.append(
            f"| {c['campaign_id']} | {c.get('status','-')} | {c.get('experiments_run',0)} | "
            f"{c.get('best_kernel_avg_us','-')} | {c.get('best_e2e_throughput','-')} | "
            f"{c.get('current_pct_of_h100',0):.2f}% |"
        )
    lines.append("")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def cmd_reset(state: dict) -> None:
    if STATE_PATH.exists():
        STATE_PATH.unlink()
    print(f"deleted {STATE_PATH}; rerun prepare.py to re-initialize")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")
    n = sub.add_parser("next")
    n.add_argument("--campaign", required=True)

    rec = sub.add_parser("recommend")
    rec.add_argument("--campaign", required=True)

    r = sub.add_parser("record")
    r.add_argument("--campaign", required=True)
    r.add_argument("--kernel-avg-us", type=float, required=True)
    r.add_argument("--e2e-throughput", type=float, required=True)
    r.add_argument("--status", choices=["keep", "revert", "crash"], required=True)
    r.add_argument("--description", required=True)

    sub.add_parser("report")
    sub.add_parser("reset")

    args = ap.parse_args()

    if args.cmd == "reset":
        cmd_reset({})
        return 0

    state = load_state()

    if args.cmd == "status":
        cmd_status(state)
    elif args.cmd == "next":
        cmd_next(state, args.campaign)
    elif args.cmd == "recommend":
        cmd_recommend(state, args.campaign)
    elif args.cmd == "record":
        cmd_record(state, campaign_name=args.campaign,
                   kernel_avg_us=args.kernel_avg_us,
                   e2e_throughput=args.e2e_throughput,
                   status=args.status, description=args.description)
    elif args.cmd == "report":
        cmd_report(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
