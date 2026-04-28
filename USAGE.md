# Usage

End-to-end guide for running an overnight autokernels-genesis session on 8× MI300X. Read the [README](README.md) first for the conceptual overview, and the [`program.md`](program.md) operating manual the agent itself reads.

## Prerequisites

- 8× MI300X box with ROCm 6.x and Docker
- `~/work/Genesis`, `~/work/quadrants`, `~/work/newton-assets` checked out on the host
- `genesis:amd-integration` Docker image built and locally available
- [`uv`](https://github.com/astral-sh/uv) on the host
- `gh` (optional) for any post-run PR work

## 1. One-time setup

Verify hardware, repos, and Docker image; calibrate the noise floor with three baseline benchmarks.

```bash
git clone https://github.com/lohiaj/autokernels-genesis.git ~/work/autokernels-genesis
cd ~/work/autokernels-genesis
uv sync

uv run prepare.py
```

`prepare.py` writes `workspace/orchestration_state.json` (per-campaign state) and `workspace/baseline_calibration.json` (sigma_pct used by the KEEP rule). Re-run any time you suspect the host changed.

## 2. Smoke test on one GPU

Sanity-check `bench.py` end-to-end before spinning up the fleet. The flags here skip everything that's slow:

```bash
HIP_VISIBLE_DEVICES=0 \
AUTOKERNEL_CONTAINER=ak-gpu0 \
AUTOKERNEL_GPU_ID=0 \
GENESIS_SRC=$HOME/work/Genesis \
AUTOKERNEL_SHARED_DIR=$HOME/work/autokernels-shared \
uv run bench.py \
  --campaign func_broad_phase \
  --skip-correctness \
  --skip-traced \
  --trials 2 \
  --no-rubric-check \
  --no-wipe-between-trials
```

Expected output (last 20 lines of run.log):

```
correctness:        SKIP
kernel_avg_us:      0.00
e2e_throughput:     343993
e2e_throughput_sigma: 312.5
e2e_throughput_sigma_pct: 0.091
e2e_throughput_n:   2
e2e_pct_of_h100:    43.31
peak_vram_mb:       284.1
e2e_throughput_samples: [344214, 343772]
```

If `e2e_pct_of_h100` is in the 30-70% band and sigma_pct is under 1%, the harness is healthy.

## 3. Spawn 8 parallel agents

```bash
launcher/launch_8gpu.sh
```

Creates per-GPU containers (`ak-gpu0` … `ak-gpu7`), per-GPU worktrees of this repo and Genesis, and the shared cross-agent log directory. Default split is 4 `func_broad_phase` + 4 `kernel_step_1_2`. Override:

```bash
launcher/launch_8gpu.sh --split 2 6   # 2 broad_phase + 6 step_1_2
launcher/launch_8gpu.sh --num-gpus 4  # only 4 GPUs
```

After this, the agents are NOT running — only the containers and worktrees. The launcher prints the exact `claude code` invocation for each GPU.

## 4. Start the watchdog

The agent loop has no programmatic stop. The watchdog is the only component allowed to set `workspace/HALT.flag`. Start it in a separate process before launching agents:

```bash
nohup uv run watchdog.py --loop --interval 300 --campaign $CAMPAIGN \
  > workspace/watchdog.log 2>&1 &
```

What it does on each pass (every 5 min by default):

- Disk: HALT at 95% used.
- Log rotation: copies `run.log` to `workspace/logs/expN.log`, keeps last 200.
- Git GC: weekly `git gc --auto` per worktree.
- Cross-agent drift: reads the shared global log; if median KEEP throughput across ≥3 GPUs has dropped > 3σ from the trailing window, sets HALT.

To clear after fixing the underlying issue:

```bash
uv run watchdog.py --clear-halt
```

## 5. Launch agents (one per GPU)

Per the launcher's printed instructions, for each GPU:

```bash
cd ~/work/ak-wt/gpu0
AUTOKERNEL_GPU_ID=0 \
AUTOKERNEL_CONTAINER=ak-gpu0 \
GENESIS_SRC=~/work/genesis-wt/gpu0 \
CAMPAIGN=func_broad_phase \
AUTOKERNEL_SHARED_DIR=~/work/autokernels-shared \
claude code
```

In the agent session, the first message is:

```
Read program.md and references/index.md, then run setup (Phase A).
When ready, enter Phase B and loop until SIGINT.
```

The agent will:

1. Read `program.md` (operating manual) and `references/` once.
2. Run Phase A: confirm assignment, create branches, init `results.tsv`, bootstrap `workspace/ideas-$CAMPAIGN.md` from playbook, run baseline.
3. Enter Phase B: the 10-step indefinite loop (read state → think → edit → commit → bench → decide → record → loop).

## 6. Watching progress overnight

```bash
# Per-GPU progress
tail -f ~/work/ak-wt/gpu0/run.log
cat ~/work/ak-wt/gpu0/results.tsv

# Cross-agent digest
uv run global_log.py digest --campaign func_broad_phase --last 100

# Per-GPU local synthesis
cat ~/work/ak-wt/gpu0/workspace/learning.md

# Watchdog log
tail -f ~/work/autokernels-genesis/workspace/watchdog.log
```

## 7. Morning gate

Run the full acceptance gate against any campaign branch when you're ready to merge:

```bash
uv run verify.py --campaign func_broad_phase > verify.log 2>&1
```

This runs the full pytest suite + the production-grade 8192/500/FP32 e2e benchmark. If it fails, `git bisect` between the campaign branch and its base to find the offending commit.

## Configuration

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `AUTOKERNEL_CONTAINER` | Docker container name to exec into | `ak-gpu$AUTOKERNEL_GPU_ID` |
| `AUTOKERNEL_GPU_ID` | GPU index pinned to this agent | `0` |
| `AUTOKERNEL_SHARED_DIR` | Cross-agent shared log directory | `$AUTOKERNEL_ROOT/autokernels-shared` |
| `GENESIS_SRC` | Host path to the Genesis worktree this agent edits | `$HOME/work/Genesis` |
| `AUTOKERNEL_ROOT` | Host root for work-tree checkouts | `$HOME/work` |
| `CAMPAIGN` | Campaign id (set by launcher) | _(required)_ |

### Tuning the KEEP rule

The 2σ KEEP rule lives in `program.md::B2`. The static noise floor and per-campaign baselines live in `workspace/baseline_calibration.json` and `workspace/orchestration_state.json` (both populated by `prepare.py`). Re-run `prepare.py` whenever you want to recalibrate.

### Adding a new campaign

1. `mkdir kernels/<new_campaign>`
2. Add `target.json` (file paths, baseline metrics, correctness pytest scope).
3. Add `playbook.md` (seed ideas).
4. `uv run prepare.py` to refresh `orchestration_state.json`.
5. `launcher/launch_8gpu.sh --split N M K` to allocate GPUs.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `bench: ABORT -- HEAD commit message is missing the B1.5 rubric` | Agent forgot the three-question forcing function in commit body | Either amend the commit with the rubric, or pass `--no-rubric-check` (only for baselines / out-of-band runs) |
| `HALT  reason=disk at NN.N% on /...` | Disk near full | Free space, then `uv run watchdog.py --clear-halt` |
| `HALT  reason=cross-agent drift: ...` | Median KEEP throughput dropped beyond noise across multiple GPUs | Re-run baseline outside the loop. If drift is real, investigate cooling / rocm version / container state |
| `bench: ERROR: container ak-gpu0 is not running` | Launcher didn't run, or container died | Re-run `launcher/launch_8gpu.sh` |
| `correctness: FAIL` on baseline | Pre-existing breakage on the Genesis branch | Stop, fix the branch state out-of-band; do not let agent iterate on a broken baseline |
| Sigma_pct > 5% on multi-trial | GPU clocks not pinned (rocm-smi unavailable, no permission) | Run container with `--privileged` or grant cap_sys_rawio |

## Retargeting to a different project

Every project- and platform-specific value lives in [`harness.toml`](harness.toml). Editing this file is sufficient to point the harness at a different ROCm project (e.g. Composable Kernel benchmarks, AITER) or a different GPU (replace clock-pin commands, profiler invocation, gpu_arch). The Python harness reads the file once at import time via `_config.py`; if the file is missing, the loader falls back to built-in Genesis-on-MI300X defaults.

What's still per-project after editing `harness.toml`:

- `kernels/<campaign>/{target.json,playbook.md}` — per-campaign edit-file scope, baselines, seed ideas.
- `references/{project_context.md,mi300x_notes.md}` — the reference docs the agent reads at setup. Replace with your project's context + your GPU's hardware notes.
- `program.md` — the operating manual. Most of it is generic (loop structure, KEEP rule, never-stop semantics); the campaign list at the top and a few project name mentions need updating.

### Per-campaign hypothesis classes

Each campaign has its own `kernels/<campaign>/classes.json` mapping hypothesis descriptions to class names. `summarize.py` and `global_log.py` use this to bucket experiments for the per-class success table. Schema:

```json
{
  "comment": "Hypothesis classification for <campaign>. ...",
  "patterns": [
    ["<class_name>", "<case-insensitive regex, first match wins>"],
    ["block_dim",   "\\bblock[_-]?dim\\b|\\bbd\\s*=\\s*\\d"],
    ["fuse",        "\\bfus(e|ed|ing)\\b|\\bmerge\\s+(loops?|kernels?)\\b"],
    ["aabb",        "\\baabb\\b|\\baxis[_-]?aligned\\b"]
  ]
}
```

If a campaign has no `classes.json`, the harness falls back to a built-in generic GPU-kernel-tuning vocabulary (block_dim, fuse, hoist, layout, async, atomic, ...) defined in `_classify.py::DEFAULT_PATTERNS`. The fallback is reasonable for any GPU project; per-campaign overrides give domain-specific signal (e.g. `aabb` and `intersect` for broad-phase, `mass_matrix` and `cartesian` for the step kernels, `constraint` and `cg_solver` for the body monolith).

What's NOT yet abstracted (Layer 3 work):

- Some Genesis-specific guidance in `program.md` (campaign names, references to `qd.kernel`/`qd.func`, the worked example uses Genesis paths). Roughly half-day to template out.

## Files the agent must NEVER modify

The harness and contract are frozen. The agent edits only files listed under `kernels/$CAMPAIGN/target.json::edit_files`. See `program.md::Hard constraints` for the full list.
