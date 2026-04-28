#!/usr/bin/env python3
"""
prepare.py -- one-time setup for autokernels-genesis.

Verifies:
  1. ROCm and gfx942 are present on the host (rocminfo).
  2. ~/work/Genesis, ~/work/quadrants, ~/work/newton-assets exist.
  3. Each MI300X GPU is idle enough to use (rocm-smi --showuse).
  4. The genesis:amd-integration docker image exists.
  5. Captures a baseline e2e throughput (3 trials) to calibrate the noise floor.
  6. Loads the hand-curated optimization plan into workspace/orchestration_state.json.

Usage:
  uv run prepare.py                 # full check + 3-trial baseline
  uv run prepare.py --skip-baseline # env check only (faster)
  uv run prepare.py --image genesis:amd-integration --container-name ak-prepare
"""

from __future__ import annotations

import argparse
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

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR / "workspace"
KERNELS = SCRIPT_DIR / "kernels"
AUTOKERNEL_ROOT = Path(os.path.expanduser("~/work"))


def step(msg: str) -> None:
    print(f"\n=== {msg} ===")

def ok(msg: str) -> None:
    print(f"  ok   {msg}")

def warn(msg: str) -> None:
    print(f"  WARN {msg}")

def fail(msg: str) -> None:
    print(f"  FAIL {msg}")
    sys.exit(1)

def run(cmd: list[str], timeout: int = 30) -> tuple[int, str]:
    # Augment PATH so /opt/rocm/bin tools work without shell init
    env = os.environ.copy()
    rocm_bin = "/opt/rocm/bin"
    if rocm_bin not in env.get("PATH", "") and os.path.isdir(rocm_bin):
        env["PATH"] = f"{rocm_bin}:{env.get('PATH', '')}"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, ""
    except FileNotFoundError:
        # Try absolute path under /opt/rocm/bin as fallback
        if not cmd[0].startswith("/") and os.path.exists(f"/opt/rocm/bin/{cmd[0]}"):
            return run([f"/opt/rocm/bin/{cmd[0]}"] + cmd[1:], timeout=timeout)
        return 127, f"{cmd[0]} not found"


# ---------------------------------------------------------------------------
# Step 1: ROCm + GPU
# ---------------------------------------------------------------------------

def check_rocm() -> dict:
    step("ROCm + GPU detection")
    rc, out = run(["rocminfo"], timeout=20)
    if rc != 0:
        fail(f"rocminfo failed (rc={rc}). Install ROCm.")
    arch = cfg.get_gpu_arch()
    if arch not in out:
        fail(f"rocminfo reports no {arch} device. (Configured GPU arch in harness.toml::host.gpu_arch)")
    ok(f"rocminfo reports {arch}")

    rc, out = run(["rocm-smi", "--showproductname"], timeout=10)
    if rc != 0:
        warn("rocm-smi not available; skipping GPU enumeration")
        return {"num_gpus": 0}
    # Count GPU lines
    n_gpus = len(re.findall(r"GPU\[\d+\]\s*:", out))
    if n_gpus == 0:
        n_gpus = out.count("MI300X")
    ok(f"detected {n_gpus} GPU(s) via rocm-smi")
    if n_gpus < 8:
        warn(f"expected 8 MI300X, found {n_gpus} -- launcher will scale to what's available")

    rc, out = run(["rocm-smi", "--showuse"], timeout=10)
    if rc == 0:
        busy = []
        for m in re.finditer(r"GPU\[(\d+)\][^\n]*?(\d+)%", out):
            gpu_id, pct = int(m.group(1)), int(m.group(2))
            if pct > 5:
                busy.append((gpu_id, pct))
        if busy:
            warn(f"GPUs in use: {busy} -- pin around them in launcher")
        else:
            ok("all GPUs idle")

    return {"num_gpus": n_gpus}


# ---------------------------------------------------------------------------
# Step 2: Repos
# ---------------------------------------------------------------------------

def check_repos() -> None:
    step("Required host directories")
    for p_str in cfg.get_required_dirs():
        p = Path(p_str)
        if not p.is_dir():
            fail(f"{p} not found (configured in harness.toml::host.required_dirs).")
        ok(f"{p} exists")

    # Genesis-specific files (URDF + benchmark_scaling.py). These live in the
    # Genesis tree itself and are project-specific, not harness-level. Soft
    # warnings only -- a harness retargeted at a non-Genesis project shouldn't
    # be blocked by missing Genesis assets.
    urdf = AUTOKERNEL_ROOT / "newton-assets" / "unitree_g1" / "urdf" / "g1_29dof.urdf"
    if urdf.exists():
        ok("g1_29dof URDF present")
    else:
        warn(f"{urdf} missing (Genesis-specific; ignore if targeting a non-Genesis project)")

    bench = AUTOKERNEL_ROOT / "Genesis" / "benchmark_scaling.py"
    if bench.exists():
        ok("Genesis/benchmark_scaling.py present")
    elif (AUTOKERNEL_ROOT / "benchmark_scaling.py").exists():
        warn(f"benchmark_scaling.py at {AUTOKERNEL_ROOT / 'benchmark_scaling.py'}, not in Genesis -- copy it in before bench")
    else:
        warn("benchmark_scaling.py not found in Genesis tree (Genesis-specific; ignore if retargeted)")


# ---------------------------------------------------------------------------
# Step 3: Docker image
# ---------------------------------------------------------------------------

def check_docker_image(image: str) -> None:
    step(f"Docker image: {image}")
    rc, out = run(["docker", "image", "inspect", image], timeout=15)
    if rc != 0:
        fail(f"image {image} not found locally. Build per prompt_mi300x.md§Build:\n"
             "  cd ~/work/quadrants && docker buildx build -f Dockerfile.rocm -t quadrants:amd-integration .\n"
             "  cd ~/work/Genesis   && docker buildx build -f Dockerfile.rocm -t genesis:amd-integration .\n"
             "OR pass --existing-container <name> to reuse a running container with Genesis+Quadrants installed.")
    ok(f"image {image} present")


def check_existing_container(name: str) -> None:
    step(f"Existing container: {name}")
    rc, out = run(["docker", "inspect", "-f", "{{.State.Running}}", name], timeout=10)
    if rc != 0 or out.strip() != "true":
        fail(f"container {name} is not running. `docker ps -a | grep {name}` to investigate.")
    ok(f"container {name} is running")
    # Sanity: can it import genesis + quadrants?
    rc, out = run(
        ["docker", "exec", name, "python3", "-c",
         "import genesis as gs; import quadrants as qd; print('genesis_ok')"],
        timeout=30,
    )
    if rc != 0 or "genesis_ok" not in out:
        fail(f"container {name} cannot import genesis+quadrants:\n{out[-500:]}")
    ok(f"container {name} can import genesis + quadrants")


# ---------------------------------------------------------------------------
# Step 4: Baseline + variance calibration
# ---------------------------------------------------------------------------

def baseline_calibration(image: str, container_name: str, n_trials: int) -> dict:
    step(f"Baseline + variance calibration ({n_trials} trials, GPU 0)")

    # Spin up a temp container
    rc, _ = run(["docker", "rm", "-f", container_name], timeout=15)  # cleanup if leftover
    cmd = [
        "docker", "run", "-d", "--rm",
        "--device=/dev/kfd", "--device=/dev/dri",
        "--group-add", "video", "--group-add", "render",
        "--security-opt", "seccomp=unconfined", "--ipc=host", "--shm-size", "32G",
        "--name", container_name,
        "-e", "HIP_VISIBLE_DEVICES=0",
        "-e", "GS_FAST_MATH=0",
        "-e", "PYOPENGL_PLATFORM=egl", "-e", "EGL_PLATFORM=surfaceless",
        "-e", "PYGLET_HEADLESS=true",
        "-v", f"{AUTOKERNEL_ROOT / 'Genesis'}:/src/Genesis",
        "-v", f"{AUTOKERNEL_ROOT / 'quadrants'}:/src/quadrants:ro",
        "-v", f"{AUTOKERNEL_ROOT / 'newton-assets'}:/src/newton-assets:ro",
        image, "sleep", "infinity",
    ]
    rc, out = run(cmd, timeout=60)
    if rc != 0:
        fail(f"docker run failed: {out[:500]}")
    ok(f"container {container_name} up")

    try:
        # Genesis sometimes wants newton-assets as a sibling symlink
        run(["docker", "exec", container_name, "bash", "-lc",
             "ln -sfn /src/newton-assets /src/Genesis/newton-assets || true"], timeout=10)

        results = []
        for trial in range(n_trials):
            print(f"  trial {trial+1}/{n_trials} (cache wipe + benchmark_scaling.py 8192/500/FP32)...")
            wipe = "rm -rf /root/.cache/quadrants /root/.cache/mesa_shader_cache 2>/dev/null; true"
            run(["docker", "exec", container_name, "bash", "-lc", wipe], timeout=30)
            t0 = time.time()
            cmd = (
                "cd /src/Genesis && "
                "python benchmark_scaling.py --precision 32 --max-envs 8192 --num-steps 500 "
                "--scaling-results-out /tmp/baseline.json 2>&1 | tail -5"
            )
            rc, out = run(["docker", "exec", container_name, "bash", "-lc", cmd], timeout=900)
            dt = time.time() - t0
            if rc != 0:
                warn(f"trial {trial+1} failed: {out[-400:]}")
                continue
            rc2, raw = run(["docker", "exec", container_name, "cat", "/tmp/baseline.json"], timeout=10)
            if rc2 != 0:
                warn(f"trial {trial+1}: could not read baseline JSON")
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                warn(f"trial {trial+1}: malformed JSON")
                continue
            entry = parsed if isinstance(parsed, dict) else (parsed[-1] if parsed else None)
            if not entry:
                continue
            thr = entry.get("throughput") or entry.get("env_steps_per_sec") or 0.0
            wall = entry.get("wall_time_s") or entry.get("wall") or dt
            ok(f"trial {trial+1}: {thr:.0f} env*steps/s, wall {wall:.2f}s")
            results.append({"throughput": float(thr), "wall_seconds": float(wall)})
    finally:
        run(["docker", "rm", "-f", container_name], timeout=20)

    if len(results) < 2:
        warn("fewer than 2 successful trials -- noise floor will default to 1.0%")
        return {"trials": results, "e2e_noise_floor_pct": 1.0,
                "median_throughput": results[0]["throughput"] if results else None}

    throughputs = [r["throughput"] for r in results]
    median = statistics.median(throughputs)
    stdev = statistics.stdev(throughputs)
    cv_pct = 100.0 * stdev / median if median else 0.0
    # Noise floor = max(1.0%, 3*sigma_pct) per design discussion
    noise_floor = max(1.0, 3.0 * cv_pct)

    print()
    print(f"  baseline median throughput: {median:.0f} env*steps/s")
    print(f"  stdev:                      {stdev:.1f}")
    print(f"  coefficient of variation:   {cv_pct:.2f}%")
    print(f"  computed noise floor:       {noise_floor:.2f}% (max of 1.0%, 3*sigma)")
    return {
        "trials": results,
        "median_throughput": median,
        "stdev": stdev,
        "cv_pct": cv_pct,
        "e2e_noise_floor_pct": noise_floor,
    }


# ---------------------------------------------------------------------------
# Step 5: Plan
# ---------------------------------------------------------------------------

def write_plan(noise_pct: float, baseline_throughput: float | None) -> None:
    step("Writing workspace/orchestration_state.json from kernel manifests")
    WORKSPACE.mkdir(exist_ok=True)
    campaigns = []
    for d in sorted(KERNELS.iterdir()):
        if not d.is_dir():
            continue
        manifest_path = d / "target.json"
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            m = json.load(f)
        campaigns.append({
            "campaign_id": m["campaign_id"],
            "current_pct_of_h100": m["current_pct_of_h100"],
            "target_pct_of_h100": m.get("target_pct_of_h100", 80.0),
            "baseline_kernel_avg_us": m.get("baseline_kernel_avg_us"),
            "current_kernel_avg_us": m.get("baseline_kernel_avg_us"),
            "best_kernel_avg_us": m.get("baseline_kernel_avg_us"),
            "best_e2e_throughput": baseline_throughput,
            "experiments_run": 0,
            "consecutive_reverts": 0,
            "status": "pending",
        })
        ok(f"loaded campaign {m['campaign_id']}")

    state = {
        "noise_floor_pct": noise_pct,
        "baseline_e2e_throughput": baseline_throughput,
        "h100_throughput_ref": cfg.get_reference_value(),
        "campaigns": campaigns,
        # Soft advisory thresholds. As of the loop redesign, these no longer
        # cause `orchestrate.py next` to return DONE -- they only emit warnings
        # on stderr. The only legitimate stop is human SIGINT or watchdog HALT.
        "move_on_criteria": {
            "warn_consecutive_reverts": 5,
            "consecutive_reverts": 8,
            "max_experiments_per_campaign": 80,
            "kernel_pct_target": 80.0,
        },
    }
    out = WORKSPACE / "orchestration_state.json"
    with open(out, "w") as f:
        json.dump(state, f, indent=2)
    ok(f"wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-baseline", action="store_true",
                    help="skip the 3-trial baseline calibration")
    ap.add_argument("--image", default=cfg.get_container_image(),
                    help="docker image to use for baseline (default from harness.toml::container.image; "
                         "ignored if --existing-container set)")
    ap.add_argument("--container-name", default="ak-prepare",
                    help="temp container name for baseline (only used when spinning a fresh image)")
    ap.add_argument("--existing-container", default=None,
                    help="reuse an already-running container with genesis+quadrants installed (e.g. 'gbench')")
    ap.add_argument("--n-trials", type=int, default=3,
                    help="number of baseline trials for variance estimation")
    args = ap.parse_args()

    check_rocm()
    check_repos()
    if args.existing_container:
        check_existing_container(args.existing_container)
    else:
        check_docker_image(args.image)

    if args.skip_baseline:
        warn("--skip-baseline: noise floor defaults to 1.0%")
        cal = {"e2e_noise_floor_pct": 1.0, "median_throughput": None}
    else:
        cal = baseline_calibration(args.image, args.container_name, args.n_trials)
        # Save calibration as its own file for the agent to read
        WORKSPACE.mkdir(exist_ok=True)
        with open(WORKSPACE / "baseline_calibration.json", "w") as f:
            json.dump(cal, f, indent=2)
        ok(f"wrote {WORKSPACE / 'baseline_calibration.json'}")

    write_plan(cal["e2e_noise_floor_pct"], cal.get("median_throughput"))

    print()
    print("Ready. Next steps:")
    print("  - Run launcher/launch_8gpu.sh to spawn 8 per-GPU containers + worktrees")
    print("  - Or smoke-test: HIP_VISIBLE_DEVICES=0 GENESIS_SRC=$HOME/work/Genesis \\")
    print("                   AUTOKERNEL_GPU_ID=0 AUTOKERNEL_CONTAINER=ak-gpu0 \\")
    print("                   uv run bench.py --campaign func_broad_phase")
    return 0


if __name__ == "__main__":
    sys.exit(main())
