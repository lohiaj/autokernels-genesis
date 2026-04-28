#!/usr/bin/env python3
"""
sandbox.py -- safe sandbox setup for autokernel single-kernel sessions.

Clones (or refreshes) Genesis + Quadrants into ~/.cache/autokernels-genesis/sandbox/
and creates a per-session branch in each, so the agent can edit freely without
touching anything outside the sandbox.

Why this exists:
  - The simple-mode loop edits source files and runs `git reset --hard` on
    revert. Doing that in a user's working repo would clobber their uncommitted
    changes. The sandbox is an isolated clone the agent owns end-to-end.
  - The user shouldn't have to clone Genesis or Quadrants by hand. This script
    pulls the latest release branch on first run and refreshes on subsequent
    runs.

Subcommands:
  uv run sandbox.py setup --kernel NAME    # ensure repos cloned + on session branch
  uv run sandbox.py path --repo Genesis    # print the sandbox path for a repo
  uv run sandbox.py path --repo Quadrants  #
  uv run sandbox.py refresh                # fetch upstream, fast-forward release branches
  uv run sandbox.py status                 # show what's in the sandbox
  uv run sandbox.py wipe --yes             # nuke the sandbox (destructive)

Env vars (with defaults):
  AUTOKERNEL_SANDBOX            ~/.cache/autokernels-genesis/sandbox
  AUTOKERNEL_GENESIS_URL        https://github.com/Genesis-Embodied-AI/Genesis
  AUTOKERNEL_GENESIS_BRANCH     main
  AUTOKERNEL_QUADRANTS_URL      (no default; if unset, Quadrants is skipped --
                                 ask your AMD team for the internal URL)
  AUTOKERNEL_QUADRANTS_BRANCH   main

The agent invokes `setup --kernel NAME` once at the start of every session.
The script is idempotent: re-running it on an existing sandbox refreshes the
release branch and creates a new session branch.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# IMPORTANT: the sandbox must live under a path that's mounted into whatever
# container you'll run the bench in. On standard AMD perf VMs the perf
# container (e.g. `gbench`) mounts $AUTOKERNEL_ROOT (default ~/work) as /work.
# Putting the sandbox under $AUTOKERNEL_ROOT keeps it visible to the bench.
# Override with $AUTOKERNEL_SANDBOX if your container mounts elsewhere.
_AUTOKERNEL_ROOT = Path(os.environ.get("AUTOKERNEL_ROOT", str(Path.home() / "work")))
DEFAULT_SANDBOX = _AUTOKERNEL_ROOT / ".cache" / "autokernels-genesis-sandbox"

# Sibling assets (typically separate repos like newton-assets) that we want
# symlinked INTO the sandbox so Genesis-style imports of e.g.
# Genesis/newton-assets/... resolve correctly inside both the host fs and
# the bench container. Keys are sandbox-relative target paths;
# values are source paths under $AUTOKERNEL_ROOT (or absolute) -- the symlink
# is created RELATIVE so it resolves in both contexts (host: /home/.../work/X,
# container: /work/X) provided both contexts share the same mount geometry.
SIBLING_ASSETS: list[tuple[str, str]] = [
    # (target inside Genesis sandbox, sibling repo name under $AUTOKERNEL_ROOT)
    ("Genesis/newton-assets", "newton-assets"),
]

REPOS: dict[str, dict[str, str]] = {
    # Both repos are public ROCm forks tuned for AMD perf work. Defaults track
    # the active AMD-perf release branch as of 2026-04 -- override via env if
    # a newer one supersedes them.
    "Genesis": {
        "url_env":    "AUTOKERNEL_GENESIS_URL",
        "url_default": "https://github.com/ROCm/Genesis.git",
        "branch_env":    "AUTOKERNEL_GENESIS_BRANCH",
        "branch_default": "release/0.4.4.amdperf",
        "required": True,
    },
    "Quadrants": {
        "url_env":    "AUTOKERNEL_QUADRANTS_URL",
        "url_default": "https://github.com/ROCm/quadrants.git",
        "branch_env":    "AUTOKERNEL_QUADRANTS_BRANCH",
        "branch_default": "amd-integration",
        "required": True,
    },
}


def _sandbox_dir() -> Path:
    return Path(os.environ.get("AUTOKERNEL_SANDBOX", str(DEFAULT_SANDBOX))).expanduser()


def _repo_meta(name: str) -> dict[str, str]:
    spec = REPOS[name]
    return {
        "url":    os.environ.get(spec["url_env"], spec["url_default"]),
        "branch": os.environ.get(spec["branch_env"], spec["branch_default"]),
        "required": spec["required"],
    }


def _run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> tuple[int, str]:
    """Run a command; on failure, print captured output. Returns (rc, stdout+stderr)."""
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if check and r.returncode != 0:
        print(f"sandbox: ERROR: command failed (rc={r.returncode}):\n  $ {' '.join(cmd)}",
              file=sys.stderr)
        print(out[-2000:], file=sys.stderr)
        sys.exit(1)
    return r.returncode, out


def _ensure_clone(name: str) -> Path | None:
    """Clone the repo into the sandbox if missing. Returns the path, or None
    if the repo is optional and no URL is configured."""
    meta = _repo_meta(name)
    if not meta["url"]:
        if meta["required"]:
            print(f"sandbox: ERROR: {name} URL not set (env {REPOS[name]['url_env']}) and "
                  f"required. Set it and rerun.", file=sys.stderr)
            sys.exit(1)
        print(f"sandbox: SKIP {name} (env {REPOS[name]['url_env']} not set; "
              f"ask your team for the URL if you need to optimize {name} kernels)")
        return None

    target = _sandbox_dir() / name
    if (target / ".git").exists():
        print(f"sandbox: {name} already cloned at {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"sandbox: cloning {name} ({meta['url']}) -> {target}")
    _run(["git", "clone", "--no-tags", meta["url"], str(target)])
    return target


def _refresh(name: str, target: Path) -> None:
    """Fetch upstream + fast-forward to the configured release branch.
    Discards any uncommitted changes in the sandbox (it's our scratch space)."""
    meta = _repo_meta(name)
    print(f"sandbox: refreshing {name} -> {meta['branch']}")
    _run(["git", "fetch", "origin", "--prune"], cwd=target)
    # Stash anything stray (sandbox is ours but be paranoid)
    _run(["git", "reset", "--hard"], cwd=target, check=False)
    _run(["git", "clean", "-fd"], cwd=target, check=False)
    _run(["git", "checkout", meta["branch"]], cwd=target)
    _run(["git", "reset", "--hard", f"origin/{meta['branch']}"], cwd=target)


def _new_session_branch(name: str, target: Path, kernel: str) -> str:
    """Create a fresh session branch off the refreshed release branch.
    Returns the branch name."""
    safe_kernel = "".join(c if c.isalnum() or c in "-_." else "-" for c in kernel)
    branch = f"autokernel/{safe_kernel}-{time.strftime('%Y%m%d-%H%M%S')}"
    _run(["git", "checkout", "-b", branch], cwd=target)
    print(f"sandbox: {name} session branch: {branch}")
    return branch


def _link_sibling_assets() -> None:
    """Create RELATIVE symlinks for sibling assets so they resolve in BOTH
    host and container contexts (provided both share the mount geometry).

    For example: ../../newton-assets from Genesis/newton-assets points to
    $SANDBOX/../newton-assets. On the host that's $AUTOKERNEL_ROOT/newton-assets;
    inside the container that's /work/newton-assets if /work is the mount
    of $AUTOKERNEL_ROOT. Both work."""
    sandbox = _sandbox_dir()
    for target_relpath, sibling in SIBLING_ASSETS:
        target = sandbox / target_relpath
        # Source: walk up from target to a sibling under $AUTOKERNEL_ROOT
        # If $AUTOKERNEL_ROOT/sibling exists, point at it via a relative path
        sibling_host = _AUTOKERNEL_ROOT / sibling
        if not sibling_host.exists():
            print(f"sandbox: SKIP symlink {target_relpath} (no {sibling_host} on host)")
            continue
        # Make sure the parent of `target` exists (it's inside a cloned repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Compute the relative path from target's parent to sibling_host
        try:
            rel = os.path.relpath(sibling_host, target.parent)
        except ValueError:
            # Different drives on Windows -- skip
            print(f"sandbox: SKIP symlink {target_relpath} (no relative path possible)")
            continue
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(rel)
        print(f"sandbox: linked {target_relpath} -> {rel}  (resolves to {sibling_host})")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_setup(args: argparse.Namespace) -> int:
    sandbox = _sandbox_dir()
    sandbox.mkdir(parents=True, exist_ok=True)
    print(f"sandbox: base = {sandbox}")
    print(f"sandbox: kernel = {args.kernel}")
    print()

    paths: dict[str, Path] = {}
    for name in REPOS:
        target = _ensure_clone(name)
        if target is None:
            continue
        _refresh(name, target)
        _new_session_branch(name, target, args.kernel)
        paths[name] = target

    # Sibling assets (newton-assets, etc.) need to be reachable from inside the
    # cloned repos so Genesis-style imports work. Use relative symlinks so they
    # resolve in both host and container.
    if paths:
        _link_sibling_assets()

    print()
    print("=" * 60)
    print("sandbox ready. Source paths the agent should grep:")
    for name, p in paths.items():
        print(f"  {name:12s} {p}")
    print("=" * 60)
    return 0


def cmd_path(args: argparse.Namespace) -> int:
    target = _sandbox_dir() / args.repo
    if not (target / ".git").exists():
        print(f"sandbox: {args.repo} not cloned yet; run `sandbox.py setup --kernel NAME` first",
              file=sys.stderr)
        sys.exit(1)
    print(target)
    return 0


def cmd_refresh(args: argparse.Namespace) -> int:
    sandbox = _sandbox_dir()
    if not sandbox.exists():
        print("sandbox: nothing to refresh (sandbox dir does not exist)", file=sys.stderr)
        return 0
    for name in REPOS:
        target = sandbox / name
        if (target / ".git").exists():
            _refresh(name, target)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    sandbox = _sandbox_dir()
    print(f"sandbox: base = {sandbox} ({'exists' if sandbox.exists() else 'missing'})")
    if not sandbox.exists():
        return 0
    for name in REPOS:
        target = sandbox / name
        if not (target / ".git").exists():
            print(f"  {name:12s} not cloned")
            continue
        rc, branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=target, check=False)
        rc2, sha = _run(["git", "rev-parse", "--short", "HEAD"], cwd=target, check=False)
        meta = _repo_meta(name)
        print(f"  {name:12s} {target}  branch={branch.strip()}  HEAD={sha.strip()}  "
              f"upstream={meta['url']} ({meta['branch']})")
    return 0


def cmd_wipe(args: argparse.Namespace) -> int:
    if not args.yes:
        print("sandbox: refusing to wipe without --yes (destructive)", file=sys.stderr)
        sys.exit(1)
    sandbox = _sandbox_dir()
    if sandbox.exists():
        shutil.rmtree(sandbox)
        print(f"sandbox: wiped {sandbox}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="clone + refresh + create session branch")
    s.add_argument("--kernel", required=True, help="kernel name (used to name the session branch)")

    p = sub.add_parser("path", help="print sandbox path for a repo")
    p.add_argument("--repo", required=True, choices=list(REPOS.keys()))

    sub.add_parser("refresh", help="fetch + fast-forward release branches (no session branch)")
    sub.add_parser("status", help="show sandbox state")

    w = sub.add_parser("wipe", help="delete the sandbox dir (destructive)")
    w.add_argument("--yes", action="store_true", help="confirm destructive wipe")

    args = ap.parse_args()
    return {
        "setup":   cmd_setup,
        "path":    cmd_path,
        "refresh": cmd_refresh,
        "status":  cmd_status,
        "wipe":    cmd_wipe,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
