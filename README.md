# autokernels-genesis

Autoresearch for Genesis-on-MI300X kernels. A targeted fork of [RightNow-AI/autokernel](https://github.com/RightNow-AI/autokernel) (which itself is the kernel-domain port of [karpathy/autoresearch](https://github.com/karpathy/autoresearch)), specialized for two Genesis kernel campaigns on 8× AMD Instinct MI300X.

## What this is

A coding agent (Claude Code, Codex) reads `program.md`, picks one of two campaigns (`func_broad_phase` or `kernel_step_1_2`), edits Genesis Python source for the targeted kernel, runs the harness on a pinned GPU inside the `genesis:amd-integration` container, parses the per-kernel rocprofv3 stats + end-to-end throughput, then keeps or reverts. Loops forever. 8 agents on 8 GPUs by default.

## Directory shape

```
autokernels-genesis/
├── program.md                          the agent's operating manual
├── project_context.md                    the project/baseline/target numbers (frozen)
├── mi300x_notes.md                     CDNA3 / gfx942 hardware cheatsheet (frozen)
├── prepare.py                          ROCm + container check, variance calibration, baseline capture
├── bench.py                            FROZEN: runs benchmark_scaling.py inside container, parses
│                                       rocprofv3 stats, prints greppable contract
├── correctness.py                      FROZEN: scoped pytest subset for each campaign
├── orchestrate.py                      decides which campaign + when to move on (Amdahl)
├── verify.py                           full acceptance gate (full pytest + 8192/500 e2e)
├── analysis.py                         morning dashboard (results.tsv → progress.png)
├── kernels/
│   ├── func_broad_phase/
│   │   ├── target.json                 file paths + line ranges + correctness scope
│   │   └── playbook.md                 ideas seeded from project tracker (TICKET)
│   └── kernel_step_1_2/
│       ├── target.json
│       └── playbook.md
├── launcher/
│   ├── launch_8gpu.sh                  spawn 8 docker containers + 8 git worktrees
│   └── docker_run.sh                   single-container template
└── workspace/                          gitignored: per-experiment runs, profiles, plan.json
```

## Quick start (host)

```bash
# 0. Verify hardware + repos exist
rocminfo | grep -E "Marketing Name:|Compute Unit:|Wavefront Size:" | head -6
ls ~/work/Genesis ~/work/quadrants ~/work/newton-assets

# 1. One-time setup (env check + baseline capture + variance calibration)
uv run prepare.py

# 2. Smoke a single campaign on GPU 0
HIP_VISIBLE_DEVICES=0 GENESIS_SRC=$HOME/work/Genesis uv run bench.py --campaign func_broad_phase

# 3. Spawn 8 parallel agents (default split: 4 broad_phase + 4 step_1_2)
launcher/launch_8gpu.sh

# 4. Then point Claude Code or Codex at the autokernels-genesis worktree for each branch:
#    cd ~/work/ak-wt/gpu0 && claude code
#    "Read program.md and project_context.md, then run setup."
```

## Constraints (hard)

1. **Edit only the files in the campaign's `target.json`.** Tests, harness, baseline numbers, benchmark_scaling.py are all frozen.
2. **Clear `/root/.cache/quadrants` and `/root/.cache/mesa_shader_cache` after every Genesis edit** (`bench.py` does this for you — never bypass).
3. **Correctness must pass** before performance is measured. `bench.py` aborts before timing if the scoped pytest subset fails.
4. **Keep on improvement only**, where "improvement" = e2e throughput up by ≥ calibrated noise floor (see `prepare.py`) AND the targeted kernel's avg_us went down. Otherwise revert.
5. **Per-GPU containers, per-GPU caches.** Never share `/root/.cache/quadrants` across containers.
6. **NEVER STOP.** The loop is autonomous. Don't ask the human "should I continue?".

## Inheritance

Code patterns adopted from `autokernel`: 5-stage correctness shape, `KERNEL_CONFIGS` per-kernel tolerance table pattern, `MOVE_ON_CRITERIA` orchestrator constants, greppable `key:` contract, TSV-per-branch + `analysis.py` morning dashboard, the Phase A/B/C `program.md` skeleton, the numbered Constraints section style.

What we replaced: profile→extract pipeline (we have hand-curated targets); generic ML kernel library (we have two physics campaigns); single-GPU loop (we have an 8× MI300X launcher); PyTorch reference oracle (we use scoped pytest + e2e regression as the oracle); NVIDIA-only optimization tiers (replaced with CDNA3/MI300X tiers in `mi300x_notes.md`).

## License

MIT (inherited from the autokernel parent).
