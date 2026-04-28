#!/usr/bin/env python3
"""
global_log.py -- shared cross-agent loss log for parallel autokernels-genesis runs.

The 8 GPU agents each work in their own worktree with their own results.tsv.
Without coordination, each agent independently re-derives that "block_dim
sweeps on file X are dead." This module gives them a single shared file they
all append to (via flock) and a digest command they read at the top of every
loop iteration.

Location: $AUTOKERNEL_SHARED_DIR/global_log.tsv
          (default: $AUTOKERNEL_ROOT/autokernels-shared/, falling back to
                    $HOME/work/autokernels-shared/)

Schema (tab-separated, header included):
  wall_ts   gpu_id   campaign   exp   commit   kernel_avg_us   e2e_throughput
  status   hypothesis_class   description

Subcommands:
  uv run global_log.py append --gpu N --campaign X --exp N --commit C \
      --kernel-avg-us K --e2e-throughput E --status keep|revert|crash \
      --description "..."
  uv run global_log.py digest --campaign X [--last 50]    -- markdown digest for prompt
  uv run global_log.py tail   --campaign X [--last 20]    -- raw rows
  uv run global_log.py path                               -- print resolved log path
  uv run global_log.py init                               -- create dir + header if missing

Locking: fcntl.flock on the file fd during append. Append-only; tail/digest
do not lock (they tolerate a partial last line).
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import io
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HEADER = [
    "wall_ts", "gpu_id", "campaign", "exp", "commit",
    "kernel_avg_us", "e2e_throughput", "status",
    "hypothesis_class", "description",
]


def _shared_dir() -> Path:
    explicit = os.environ.get("AUTOKERNEL_SHARED_DIR")
    if explicit:
        return Path(explicit)
    workspace_root = os.environ.get("AUTOKERNEL_ROOT") or os.path.expanduser("~/work")
    return Path(workspace_root) / "autokernels-shared"


def log_path() -> Path:
    return _shared_dir() / "global_log.tsv"


def ensure_log() -> Path:
    p = log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists() or p.stat().st_size == 0:
        with open(p, "w") as f:
            f.write("\t".join(HEADER) + "\n")
    return p


# ---------------------------------------------------------------------------
# Hypothesis classification (mirrors summarize.py's CLASS_PATTERNS)
# ---------------------------------------------------------------------------

CLASS_PATTERNS = [
    ("block_dim",   re.compile(r"\bblock[_-]?dim\b|\bbd\s*=\s*\d|\bblock\s*size\b", re.I)),
    ("fuse",        re.compile(r"\bfus(e|ed|ing)\b|\bmerge\s+(loops?|kernels?)\b", re.I)),
    ("hoist",       re.compile(r"\bhoist(ed|ing)?\b|\bcommon\s+sub|\binvariant\b", re.I)),
    ("inline",      re.compile(r"\binlin(e|ed|ing)\b", re.I)),
    ("atomic",      re.compile(r"\batomic", re.I)),
    ("swap",        re.compile(r"\bswap\b|\breorder\b", re.I)),
    ("simplify",    re.compile(r"\bsimplif|\bdelete|\bdedup|\bremove\s+(redundant|unused)|\bO\(1\)", re.I)),
    ("async",       re.compile(r"\basync\b|\bglobal[_-]?load[_-]?lds\b|\boverlap\b", re.I)),
    ("prefetch",    re.compile(r"\bprefetch", re.I)),
    ("layout",      re.compile(r"\b(soa|aos)\b|\blayout\b|\bvec3\b|\bbitfield\b|\bpack\b", re.I)),
    ("scheduling",  re.compile(r"\boccupancy\b|\bwave(s|fronts?)?\b|\bxcd\b|\bpersistent\b|\bstream[_-]?k\b", re.I)),
    ("memory",      re.compile(r"\blds\b|\bbank\s+conflict\b|\bvgpr\b|\bagpr\b|\bregister\b|\bspill\b", re.I)),
    ("algorithm",   re.compile(r"\balgorithm\b|\bdecomp(ose|osition)?\b|\brefactor\b|\bstructur", re.I)),
]


def classify(description: str) -> str:
    if not description:
        return "unknown"
    for name, rx in CLASS_PATTERNS:
        if rx.search(description):
            return name
    return "other"


# ---------------------------------------------------------------------------
# Append
# ---------------------------------------------------------------------------

def cmd_append(args: argparse.Namespace) -> int:
    p = ensure_log()
    cls = args.hypothesis_class or classify(args.description)
    row = [
        datetime.utcnow().isoformat() + "Z",
        str(args.gpu),
        args.campaign,
        str(args.exp),
        args.commit,
        f"{args.kernel_avg_us:.2f}",
        f"{args.e2e_throughput:.0f}",
        args.status,
        cls,
        args.description.replace("\t", " ").replace("\n", " "),
    ]
    line = "\t".join(row) + "\n"
    # Open in append mode + flock so concurrent agents serialize cleanly
    with open(p, "a") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
    print(f"global_log: appended row gpu={args.gpu} exp={args.exp} class={cls}")
    return 0


# ---------------------------------------------------------------------------
# Read / digest
# ---------------------------------------------------------------------------

def _read_rows() -> list[dict]:
    p = log_path()
    if not p.exists():
        return []
    try:
        # Read with no lock; tolerate truncated final line by ignoring DictReader errors
        text = p.read_text()
    except OSError:
        return []
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def _filter(rows: list[dict], campaign: str | None, last: int) -> list[dict]:
    if campaign:
        rows = [r for r in rows if r.get("campaign") == campaign]
    if last and last > 0:
        rows = rows[-last:]
    return rows


def cmd_tail(args: argparse.Namespace) -> int:
    rows = _filter(_read_rows(), args.campaign, args.last)
    if not rows:
        print("(empty)")
        return 0
    cols = ["gpu_id", "exp", "status", "hypothesis_class", "kernel_avg_us",
            "e2e_throughput", "description"]
    for r in rows:
        print("\t".join(str(r.get(c, ""))[:80] for c in cols))
    return 0


def cmd_digest(args: argparse.Namespace) -> int:
    rows = _filter(_read_rows(), args.campaign, args.last)
    if not rows:
        print(f"# global digest ({args.campaign or 'all'})\n\n_(no shared rows yet)_")
        return 0

    by_gpu: dict[str, dict[str, int]] = defaultdict(lambda: {"keep": 0, "revert": 0, "crash": 0})
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"keep": 0, "revert": 0, "crash": 0})
    for r in rows:
        st = (r.get("status") or "").lower()
        if st not in ("keep", "revert", "crash"):
            continue
        by_gpu[r.get("gpu_id", "?")][st] += 1
        by_class[r.get("hypothesis_class", "?")][st] += 1

    out: list[str] = []
    out.append(f"# global digest -- campaign={args.campaign or 'all'}, last {len(rows)} rows")
    out.append("")
    out.append("## Per-class success across all GPUs")
    out.append("")
    out.append("| class | keep | revert | crash | total | success |")
    out.append("|---|---|---|---|---|---|")
    for c in sorted(by_class, key=lambda k: -(by_class[k]["keep"] + by_class[k]["revert"])):
        s = by_class[c]
        tot = s["keep"] + s["revert"] + s["crash"]
        rate = (100 * s["keep"] / tot) if tot else 0
        out.append(f"| {c} | {s['keep']} | {s['revert']} | {s['crash']} | {tot} | {rate:.0f}% |")

    out.append("")
    out.append("## Per-GPU activity")
    out.append("")
    out.append("| gpu | keep | revert | crash |")
    out.append("|---|---|---|---|")
    for g in sorted(by_gpu):
        s = by_gpu[g]
        out.append(f"| {g} | {s['keep']} | {s['revert']} | {s['crash']} |")

    out.append("")
    out.append("## Recent KEEPs (cross-agent -- consider replicating these on your branch)")
    out.append("")
    keeps = [r for r in rows if (r.get("status") or "").lower() == "keep"][-10:]
    if keeps:
        for r in keeps:
            desc = (r.get("description") or "")[:140]
            out.append(f"- gpu{r.get('gpu_id')} exp{r.get('exp')} "
                       f"[{r.get('hypothesis_class')}]: {desc}")
    else:
        out.append("_(no keeps yet)_")

    out.append("")
    out.append("## Dead-end classes across all GPUs (>=4 attempts, 0 keeps)")
    out.append("")
    dead = [c for c, s in by_class.items()
            if (s["keep"] + s["revert"] + s["crash"]) >= 4 and s["keep"] == 0]
    if dead:
        for c in dead:
            s = by_class[c]
            tot = s["keep"] + s["revert"] + s["crash"]
            out.append(f"- **{c}** ({tot} attempts, 0 keeps) -- DO NOT propose this class")
    else:
        out.append("_(none yet)_")

    print("\n".join(out))
    return 0


def cmd_path(args: argparse.Namespace) -> int:
    print(log_path())
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    p = ensure_log()
    print(f"global_log: ready at {p}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("append")
    a.add_argument("--gpu", required=True, type=int)
    a.add_argument("--campaign", required=True)
    a.add_argument("--exp", required=True, type=int)
    a.add_argument("--commit", required=True)
    a.add_argument("--kernel-avg-us", required=True, type=float)
    a.add_argument("--e2e-throughput", required=True, type=float)
    a.add_argument("--status", required=True, choices=["keep", "revert", "crash"])
    a.add_argument("--hypothesis-class", default=None,
                   help="optional override; otherwise inferred from --description")
    a.add_argument("--description", required=True)

    d = sub.add_parser("digest")
    d.add_argument("--campaign", default=None)
    d.add_argument("--last", type=int, default=50)

    t = sub.add_parser("tail")
    t.add_argument("--campaign", default=None)
    t.add_argument("--last", type=int, default=20)

    sub.add_parser("path")
    sub.add_parser("init")

    args = ap.parse_args()
    return {
        "append": cmd_append,
        "digest": cmd_digest,
        "tail":   cmd_tail,
        "path":   cmd_path,
        "init":   cmd_init,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
