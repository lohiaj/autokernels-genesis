#!/usr/bin/env python3
"""
summarize.py -- distill results.tsv into workspace/learning.md.

Run after every `orchestrate.py record`. Reads results.tsv, classifies each
experiment description into a coarse hypothesis class (block_dim, fuse, hoist,
inline, atomic, swap, simplify, async, prefetch, layout, ...), and writes:

  workspace/learning.md
    - per-class success rate (kept vs reverted vs crashed)
    - per-file success rate
    - last 5 reverts with parsed "why"
    - dead-end list (classes with >=3 attempts and 0 keeps)

Also bootstraps workspace/ideas.md from each campaign's playbook on first run,
so the agent has a single editable scratchpad to mark used / append derived.

Usage:
  uv run summarize.py                       # process all campaigns in results.tsv
  uv run summarize.py --campaign func_broad_phase   # only one
  uv run summarize.py --bootstrap-ideas             # ensure ideas.md exists per campaign
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

from _classify import classify

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR / "workspace"
KERNELS = SCRIPT_DIR / "kernels"
RESULTS_TSV = SCRIPT_DIR / "results.tsv"


# ---------------------------------------------------------------------------
# Outcome inference (from description, since results.tsv has no explicit status col)
# ---------------------------------------------------------------------------

KEPT_RE = re.compile(r"\b(KEPT|KEEP|kept)\b")
REVERT_RE = re.compile(r"\b(REVERTED|REVERT|reverted|discard)\b", re.I)
CRASH_RE = re.compile(r"\b(CRASH|crash|TIMEOUT|FAIL|ERROR)\b")


def outcome(row: dict) -> str:
    # Prefer explicit correctness column for crashes
    correctness = (row.get("correctness") or "").strip().upper()
    if correctness in ("FAIL", "CRASH", "TIMEOUT"):
        return "crash"

    desc = row.get("description", "") or ""
    if "baseline" in desc.lower():
        return "baseline"
    if KEPT_RE.search(desc):
        return "kept"
    if REVERT_RE.search(desc):
        return "reverted"
    if CRASH_RE.search(desc):
        return "crash"
    # Default: if no explicit marker, treat as kept (agent forgot to annotate)
    return "kept"


def edited_file(row: dict, manifest_files: list[str]) -> str:
    """Best-effort: extract a short file token from the description."""
    desc = row.get("description", "") or ""
    for f in manifest_files:
        # Match the basename
        base = f.rsplit("/", 1)[-1]
        if base and base in desc:
            return base
    # Fall back to any *.py token
    m = re.search(r"\b([\w]+\.py)\b", desc)
    return m.group(1) if m else "?"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_results() -> list[dict]:
    if not RESULTS_TSV.exists():
        return []
    with open(RESULTS_TSV) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return rows


def load_campaign_files(campaign: str) -> list[str]:
    target = KERNELS / campaign / "target.json"
    if not target.exists():
        return []
    import json
    try:
        m = json.loads(target.read_text())
    except (OSError, ValueError):
        return []
    return [ef.get("path", "") for ef in m.get("edit_files", []) if isinstance(ef, dict)]


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render_class_table(stats: dict[str, dict[str, int]]) -> str:
    lines = ["| class | attempts | kept | reverted | crash | success_rate |",
             "|---|---|---|---|---|---|"]
    for cls in sorted(stats, key=lambda k: -stats[k]["attempts"]):
        s = stats[cls]
        att = s["attempts"]
        kept = s["kept"]
        rate = (100.0 * kept / att) if att else 0.0
        lines.append(f"| {cls} | {att} | {kept} | {s['reverted']} | {s['crash']} | {rate:.0f}% |")
    return "\n".join(lines)


def render_file_table(stats: dict[str, dict[str, int]]) -> str:
    lines = ["| file | attempts | kept | reverted | crash |",
             "|---|---|---|---|---|"]
    for f in sorted(stats, key=lambda k: -stats[k]["attempts"]):
        s = stats[f]
        lines.append(f"| {f} | {s['attempts']} | {s['kept']} | {s['reverted']} | {s['crash']} |")
    return "\n".join(lines)


def render_recent_reverts(rows: list[dict], n: int = 5) -> str:
    reverts = [r for r in rows if outcome(r) == "reverted"][-n:]
    if not reverts:
        return "_(no reverts yet)_"
    lines = []
    for r in reverts:
        exp = r.get("experiment", "?")
        cls = classify(r.get("description", ""), r.get("campaign"))
        desc = (r.get("description") or "").replace("\n", " ")[:140]
        lines.append(f"- exp{exp} [{cls}]: {desc}")
    return "\n".join(lines)


def render_dead_ends(class_stats: dict[str, dict[str, int]], min_attempts: int = 3) -> str:
    dead = [c for c, s in class_stats.items()
            if s["attempts"] >= min_attempts and s["kept"] == 0]
    if not dead:
        return "_(none yet -- keep exploring)_"
    return "\n".join(f"- **{c}** ({class_stats[c]['attempts']} attempts, 0 keeps)" for c in dead)


def summarize_campaign(campaign: str, all_rows: list[dict]) -> str:
    rows = [r for r in all_rows if (r.get("campaign") or "") == campaign]
    if not rows:
        return f"# {campaign}\n\n_(no experiments yet)_\n"

    class_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"attempts": 0, "kept": 0, "reverted": 0, "crash": 0})
    file_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"attempts": 0, "kept": 0, "reverted": 0, "crash": 0})

    manifest_files = load_campaign_files(campaign)
    n_kept = n_reverted = n_crash = 0

    for r in rows:
        oc = outcome(r)
        if oc == "baseline":
            continue
        cls = classify(r.get("description", ""), campaign)
        f = edited_file(r, manifest_files)
        class_stats[cls]["attempts"] += 1
        file_stats[f]["attempts"] += 1
        if oc in ("kept", "reverted", "crash"):
            class_stats[cls][oc] += 1
            file_stats[f][oc] += 1
        if oc == "kept":
            n_kept += 1
        elif oc == "reverted":
            n_reverted += 1
        elif oc == "crash":
            n_crash += 1

    out = []
    out.append(f"# {campaign}")
    out.append("")
    out.append(f"experiments: {len(rows)} | kept: {n_kept} | reverted: {n_reverted} | crash: {n_crash}")
    out.append("")
    out.append("## Per-hypothesis-class success")
    out.append("")
    out.append(render_class_table(class_stats))
    out.append("")
    out.append("## Per-file activity")
    out.append("")
    out.append(render_file_table(file_stats))
    out.append("")
    out.append("## Dead-end classes (>=3 attempts, 0 keeps)")
    out.append("")
    out.append(render_dead_ends(class_stats))
    out.append("")
    out.append("## Last 5 reverts")
    out.append("")
    out.append(render_recent_reverts(rows))
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# ideas.md bootstrap
# ---------------------------------------------------------------------------

IDEAS_HEADER = """# {campaign} — live ideas pool

This file is your scratchpad. The seed below was copied from
`kernels/{campaign}/playbook.md`. As you work:

- mark a tried idea inline:  `- [x] <idea>  -- exp14 KEPT (kernel 449->412us)`
- append derived ideas at the bottom under "## Derived"
- delete ideas that are clearly dead (note them in workspace/learning.md)

The playbook stays as immutable seed. This file IS the working set.

---

## Seed (from playbook.md)

"""


def bootstrap_ideas(campaign: str) -> Path | None:
    pb = KERNELS / campaign / "playbook.md"
    out = WORKSPACE / f"ideas-{campaign}.md"
    if out.exists():
        return None
    WORKSPACE.mkdir(exist_ok=True)
    seed = pb.read_text() if pb.exists() else "_(no playbook.md found)_\n"
    out.write_text(IDEAS_HEADER.format(campaign=campaign) + seed + "\n\n## Derived\n\n_(append your own here)_\n")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_campaigns(rows: list[dict]) -> list[str]:
    out: list[str] = []
    seen = set()
    for r in rows:
        c = r.get("campaign")
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    # Also include any campaign with a target.json even if no rows yet
    if KERNELS.exists():
        for d in sorted(KERNELS.iterdir()):
            if d.is_dir() and (d / "target.json").exists() and d.name not in seen:
                out.append(d.name)
                seen.add(d.name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default=None,
                    help="restrict summary to one campaign (default: all)")
    ap.add_argument("--bootstrap-ideas", action="store_true",
                    help="also create workspace/ideas-<campaign>.md from playbook if missing")
    ap.add_argument("--out", default=str(WORKSPACE / "learning.md"),
                    help="output path for the learning summary")
    args = ap.parse_args()

    rows = load_results()
    campaigns = [args.campaign] if args.campaign else discover_campaigns(rows)
    if not campaigns:
        print("summarize: no campaigns found", file=sys.stderr)
        return 1

    sections = []
    sections.append("# autokernels-genesis -- learning summary")
    sections.append("")
    sections.append(f"_(distilled from {RESULTS_TSV.name}; {len(rows)} total rows)_")
    sections.append("")
    sections.append("Read this at the top of every loop iteration. The classes with low success rates")
    sections.append("are dead ends -- switch class. The classes with high success rates show what")
    sections.append("works on this campaign -- propose more of those.")
    sections.append("")

    for c in campaigns:
        sections.append(summarize_campaign(c, rows))
        if args.bootstrap_ideas:
            created = bootstrap_ideas(c)
            if created:
                print(f"summarize: bootstrapped {created}")

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(sections) + "\n")
    print(f"summarize: wrote {out} ({len(campaigns)} campaign(s), {len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
