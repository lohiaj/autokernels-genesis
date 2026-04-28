# autokernels-genesis

Autonomous LLM-driven research loop for AMD GPU kernel optimization. Targets the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) physics simulator on 8× AMD Instinct MI300X.

A coding agent (Claude Code, Codex) reads `program.md`, picks a kernel campaign, edits Genesis Python source, runs a multi-trial benchmark inside a pinned-GPU container, parses per-kernel rocprofv3 stats and end-to-end throughput, then keeps the change only if it survives a 2-sigma improvement test. Loops indefinitely. Eight agents work in parallel, sharing a cross-agent loss log so they avoid duplicate dead-ends.

## Highlights

- **Indefinite autonomous loop.** No code-level stop condition; only human SIGINT or watchdog HALT exits.
- **Multi-trial 2-sigma KEEP rule.** Three trials per experiment with sigma reporting, paired with GPU clock pinning to halve inter-trial variance. KEEP rule is disjunctive — any single significant signal qualifies.
- **B1.5 forcing-function rubric.** Every hypothesis commit must include three lines: dominant bottleneck, smallest change, prior probability. `bench.py` enforces the rubric at the gate.
- **Cross-agent shared learning.** All 8 agents append to a flock-locked global log; each iteration reads a digest of recent KEEPs and dead-end classes from siblings.
- **Per-class success memory.** `summarize.py` distills `results.tsv` into `workspace/learning.md` so the next iteration sees what's worked and what hasn't.
- **Cross-agent statistical drift HALT.** Watchdog detects infrastructure drift by monitoring KEEP-throughput trends across all GPUs.
- **Single-file config.** Every project- and platform-specific value (reference baseline, container image, bench script, profiler command, GPU clock-pin command, cache wipe paths) is in [`harness.toml`](harness.toml). Edit one file to retarget the harness at a different ROCm project or GPU.

## Architecture

```
autokernels-genesis/
├── program.md                  agent operating manual (the prompt)
├── harness.toml                single-file config: reference, bench, container, profiler, GPU
├── _config.py                  TOML loader with safe fallbacks
├── prepare.py                  one-time env check + variance calibration
├── bench.py                    multi-trial bench + sigma + clock pin + rubric gate
├── correctness.py              scoped pytest subset per campaign
├── verify.py                   full acceptance gate (full pytest + e2e)
├── orchestrate.py              advisory campaign-state tracker
├── summarize.py                results.tsv -> workspace/learning.md
├── global_log.py               cross-agent shared loss log
├── watchdog.py                 indefinite-run safety daemon
├── analysis.py                 morning dashboard (results.tsv -> progress.png)
├── references/                 frozen reference material (consult on demand)
│   ├── index.md
│   ├── project_context.md        baselines, top-8 kernel gap, shipped patterns
│   └── mi300x_notes.md         CDNA3 / gfx942 hardware cheatsheet
├── kernels/                    per-campaign manifests
│   ├── func_broad_phase/
│   ├── kernel_step_1_2/
│   └── func_solve_body_monolith/
├── launcher/
│   ├── launch_8gpu.sh          spawn 8 docker containers + worktrees
│   └── docker_run.sh           single-container template
└── workspace/                  gitignored runtime artifacts
```

## Requirements

- Linux x86_64
- Docker 20+
- ROCm 6.x with `rocminfo`, `rocm-smi`, `rocprofv3`
- 8× AMD Instinct MI300X (or 1× for single-GPU smoke)
- Python 3.10+ with [`uv`](https://github.com/astral-sh/uv)
- A `genesis:amd-integration` Docker image with Genesis + Quadrants installed
- Local checkouts of [Genesis](https://github.com/Genesis-Embodied-AI/Genesis), Quadrants, newton-assets at `~/work/`

## Installation

```bash
git clone https://github.com/lohiaj/autokernels-genesis.git
cd autokernels-genesis
uv sync
```

## Usage

See [USAGE.md](USAGE.md) for the full agent workflow. Quick start:

```bash
# 1. One-time environment check + variance calibration
uv run prepare.py

# 2. Single-GPU smoke against an existing container
HIP_VISIBLE_DEVICES=0 \
AUTOKERNEL_CONTAINER=ak-gpu0 \
GENESIS_SRC=$HOME/work/Genesis \
uv run bench.py --campaign func_broad_phase --no-rubric-check

# 3. Spawn 8 parallel agents (default: 4 broad_phase + 4 step_1_2)
launcher/launch_8gpu.sh

# 4. Start the watchdog (separate process)
nohup uv run watchdog.py --loop --interval 300 --campaign $CAMPAIGN \
  > workspace/watchdog.log 2>&1 &

# 5. Point an agent at each worktree (one per GPU):
cd ~/work/ak-wt/gpu0 && claude code  # then: "Read program.md, then run setup."
```

## How the loop works

Each agent runs a 10-step cycle until SIGINT or HALT:

1. Check `workspace/HALT.flag`; exit cleanly if present.
2. Read state: `learning.md`, `ideas-$CAMPAIGN.md`, cross-agent digest.
3. Form one focused hypothesis (must answer the B1.5 three-question rubric).
4. Edit one Genesis file from `target.json::edit_files`.
5. Commit with the rubric in the message body.
6. Run `bench.py --trials 3` (multi-trial untraced + traced rocprofv3, GPU clocks pinned).
7. Parse the contract.
8. Decide via B2 disjunctive 2σ KEEP rule.
9. Record locally + globally; refresh `learning.md`.
10. Loop.

Full prompt (the operating manual the agent reads) is in [`program.md`](program.md).

## Output

After an overnight run the agent produces:

- `results.tsv` — every experiment with metrics and status (30-100 rows per GPU)
- A campaign branch (`perf/jlohia/$TAG`) on the Genesis repo with kept commits stacked
- `workspace/learning.md` — per-class success rates and dead-end list
- `workspace/compiler_proposals.md` — compiler-layer hypotheses for human review

## License

MIT — see [LICENSE](LICENSE).
