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
  uv run sandbox.py setup --kernel NAME    # clone repos + session branch + (if container) pip-swap
  uv run sandbox.py teardown               # restore pip + clear session state (call at end of session)
  uv run sandbox.py verify                 # check sandbox hasn't been tampered with by another agent
  uv run sandbox.py path --repo Genesis    # print the sandbox path for a repo
  uv run sandbox.py path --repo Quadrants  #
  uv run sandbox.py refresh                # fetch upstream, fast-forward release branches
  uv run sandbox.py status                 # show what's in the sandbox
  uv run sandbox.py wipe --yes             # nuke the sandbox (destructive)

Bug 1 (pip-editable swap): on standard AMD perf VMs, the perf container has
genesis-world installed pip-editable pointing at the user's main /work/Genesis.
PYTHONPATH overrides don't beat editable installs, so the bench would silently
run the user's main checkout while the agent edited the sandbox. `setup` now
detects this and swaps the editable install to the sandbox; `teardown` restores.

Bug 3 (sandbox tamper detection): on multi-tenant perf VMs, other agents' git
operations can wander into our sandbox (e.g. `git -c safe.directory=*` checkout).
`setup` writes a session state file recording the expected branch + pid + ts;
`verify` checks the sandbox is still on the expected branch. bench.py calls
`verify` before each bench and halts on mismatch.

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
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
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


# ---------------------------------------------------------------------------
# Bug 1 fix: pip-editable swap
# ---------------------------------------------------------------------------
# On standard AMD perf VMs, the perf container has genesis-world installed
# pip-editable pointing at the user's main /work/Genesis. PYTHONPATH overrides
# DO NOT beat editable installs (Python's site-packages comes first via .pth
# files). So the bench silently runs the user's main checkout while the agent
# happily edits the sandbox. Detect + swap on setup; restore on teardown.

_PIP_PACKAGE = "genesis-world"  # what's installed via pip
_SESSION_STATE_FILE = "session_state.json"  # in sandbox dir


def _container_name() -> str | None:
    """The perf container the bench runs in. None if not configured (e.g.
    non-Genesis project; pip-swap is a no-op then)."""
    return os.environ.get("AUTOKERNEL_CONTAINER")


def _container_path_for(host_path: Path) -> str:
    """Translate host path -> in-container path under the standard $AUTOKERNEL_ROOT
    -> /work mount."""
    rel = host_path.relative_to(_AUTOKERNEL_ROOT)
    return f"/work/{rel}"


def _docker_exec(container: str, *cmd: str, timeout: int = 60) -> tuple[int, str]:
    """Run command inside container; return (rc, combined_output)."""
    try:
        r = subprocess.run(
            ["docker", "exec", container, *cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except FileNotFoundError:
        return 127, "docker not found"


def _detect_pip_editable(container: str, package: str = _PIP_PACKAGE) -> str | None:
    """Return the editable install location of `package` inside `container`,
    or None if not editable / not installed."""
    rc, out = _docker_exec(container, "pip", "show", package, timeout=20)
    if rc != 0:
        return None
    # Look for "Editable project location: /path"
    for line in out.splitlines():
        if line.startswith("Editable project location:"):
            return line.split(":", 1)[1].strip()
    return None


def _pip_install_editable(container: str, path_in_container: str) -> tuple[int, str]:
    """`pip install -e PATH --no-deps --no-build-isolation` inside container."""
    return _docker_exec(
        container, "pip", "install", "-e", path_in_container,
        "--no-deps", "--no-build-isolation",
        timeout=180,
    )


def _do_pip_swap(sandbox_genesis_host: Path) -> dict:
    """Swap the container's editable genesis-world install to point at the
    sandbox. Returns a state dict to be persisted for teardown.

    No-op (returns {}) if:
      - no container configured
      - genesis-world not installed in the container
      - genesis-world not editable in the container
      - editable install already points at the sandbox
    """
    container = _container_name()
    if not container:
        return {}

    sandbox_in_container = _container_path_for(sandbox_genesis_host)
    original = _detect_pip_editable(container)
    if not original:
        print(f"sandbox: pip-swap SKIP -- {_PIP_PACKAGE} not editable in {container}")
        return {}
    if original.rstrip("/") == sandbox_in_container.rstrip("/"):
        print(f"sandbox: pip-swap SKIP -- {_PIP_PACKAGE} already points at sandbox")
        return {"container": container, "original": original, "already_swapped": True}

    print(f"sandbox: pip-swap -- {original} -> {sandbox_in_container} (in {container})")
    rc, out = _pip_install_editable(container, sandbox_in_container)
    if rc != 0:
        print(f"sandbox: WARN: pip-swap failed (rc={rc}): {out[-500:]}", file=sys.stderr)
        print(f"sandbox: WARN: bench will continue using ORIGINAL pip install -- "
              f"your edits may not take effect!", file=sys.stderr)
        return {}
    return {"container": container, "original": original, "swapped_to": sandbox_in_container}


def _do_pip_restore(state: dict) -> None:
    """Restore the editable install to its pre-swap location."""
    if not state or state.get("already_swapped"):
        return
    container = state.get("container")
    original = state.get("original")
    if not (container and original):
        return
    print(f"sandbox: pip-restore -- restoring {_PIP_PACKAGE} editable to {original}")
    rc, out = _pip_install_editable(container, original)
    if rc != 0:
        print(f"sandbox: WARN: pip-restore failed (rc={rc}): {out[-500:]}", file=sys.stderr)
        print(f"sandbox: WARN: container's {_PIP_PACKAGE} install may be stuck "
              f"pointing at the sandbox; manually re-run: "
              f"docker exec {container} pip install -e {original} --no-deps --no-build-isolation",
              file=sys.stderr)


# ---------------------------------------------------------------------------
# Bug 3 fix: session state + tamper detection
# ---------------------------------------------------------------------------

def _session_state_path() -> Path:
    return _sandbox_dir() / _SESSION_STATE_FILE


def _write_session_state(branches: dict[str, str], pip_swap: dict) -> None:
    """Record session metadata so we can detect tampering later."""
    state = {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "started_at": datetime.utcnow().isoformat() + "Z",
        "host": socket.gethostname(),
        "user": os.environ.get("USER", "?"),
        "container": _container_name(),
        "branches": branches,    # {repo_name: expected_branch}
        "pip_swap": pip_swap,    # state for restore (or {})
    }
    p = _session_state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2) + "\n")


def _read_session_state() -> dict | None:
    p = _session_state_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _verify_session_intact() -> tuple[bool, str]:
    """Check that each cloned repo is still on its expected session branch.
    Returns (ok, reason). reason is empty if ok."""
    state = _read_session_state()
    if not state:
        return False, "no session_state.json (sandbox may have been wiped)"
    sandbox = _sandbox_dir()
    for name, expected in (state.get("branches") or {}).items():
        repo = sandbox / name
        if not (repo / ".git").exists():
            return False, f"{name} repo missing or .git removed (was at {repo})"
        rc, current = _run(["git", "-C", str(repo), "branch", "--show-current"], check=False)
        if rc != 0:
            return False, f"{name}: git branch --show-current failed"
        current = current.strip()
        if current != expected:
            return False, (f"{name}: on branch '{current}' but expected '{expected}'  "
                           f"(another process may have run git checkout in your sandbox -- "
                           f"check `git -C {repo} reflog`)")
    return True, ""


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
    branches: dict[str, str] = {}
    for name in REPOS:
        target = _ensure_clone(name)
        if target is None:
            continue
        _refresh(name, target)
        branch = _new_session_branch(name, target, args.kernel)
        paths[name] = target
        branches[name] = branch

    # Sibling assets (newton-assets, etc.) need to be reachable from inside the
    # cloned repos so Genesis-style imports work. Use relative symlinks so they
    # resolve in both host and container.
    if paths:
        _link_sibling_assets()

    # Bug 1 fix: swap the container's pip-editable Genesis to point at the
    # sandbox so the bench actually runs the agent's edits.
    pip_swap = {}
    if "Genesis" in paths:
        pip_swap = _do_pip_swap(paths["Genesis"])

    # Bug 3 fix: persist session state so we can detect mid-session tampering.
    _write_session_state(branches, pip_swap)

    print()
    print("=" * 60)
    print("sandbox ready. Source paths the agent should grep:")
    for name, p in paths.items():
        print(f"  {name:12s} {p}  (branch: {branches[name]})")
    print("=" * 60)
    if pip_swap.get("original"):
        print(f"NOTE pip-swap active. Run `uv run sandbox.py teardown` at end of session "
              f"to restore the container's editable install.")
    print("=" * 60)
    return 0


def cmd_teardown(args: argparse.Namespace) -> int:
    """Restore the container's pip-editable install + clear session state.
    Idempotent: safe to call multiple times or when no session is active.
    Does NOT delete the sandbox dirs (use `wipe --yes` for that)."""
    state = _read_session_state()
    if not state:
        print("sandbox: teardown -- no active session (nothing to restore)")
        return 0
    pip_swap = state.get("pip_swap") or {}
    _do_pip_restore(pip_swap)
    _session_state_path().unlink(missing_ok=True)
    print("sandbox: teardown complete; session state cleared")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Check that the sandbox is still on its expected branches. Exits 2 on
    tamper (so callers can `set -e` style propagate the failure)."""
    ok, reason = _verify_session_intact()
    if ok:
        state = _read_session_state() or {}
        for name, branch in (state.get("branches") or {}).items():
            print(f"sandbox: {name} -- on {branch} OK")
        return 0
    print(f"sandbox: TAMPER DETECTED -- {reason}", file=sys.stderr)
    print(f"sandbox: hint: another agent may have run git operations in this sandbox.",
          file=sys.stderr)
    print(f"sandbox: hint: try `git -C $REPO reflog` to recover; then re-run setup.",
          file=sys.stderr)
    sys.exit(2)


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

    s = sub.add_parser("setup", help="clone + refresh + create session branch + pip-swap")
    s.add_argument("--kernel", required=True, help="kernel name (used to name the session branch)")

    sub.add_parser("teardown",
                   help="restore container's pip-editable install + clear session state")
    sub.add_parser("verify",
                   help="check sandbox hasn't been tampered with (exits 2 on tamper)")

    p = sub.add_parser("path", help="print sandbox path for a repo")
    p.add_argument("--repo", required=True, choices=list(REPOS.keys()))

    sub.add_parser("refresh", help="fetch + fast-forward release branches (no session branch)")
    sub.add_parser("status", help="show sandbox state")

    w = sub.add_parser("wipe", help="delete the sandbox dir (destructive)")
    w.add_argument("--yes", action="store_true", help="confirm destructive wipe")

    args = ap.parse_args()
    return {
        "setup":    cmd_setup,
        "teardown": cmd_teardown,
        "verify":   cmd_verify,
        "path":     cmd_path,
        "refresh":  cmd_refresh,
        "status":   cmd_status,
        "wipe":     cmd_wipe,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
