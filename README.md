# autokernels

Autonomous GPU-kernel optimization loop. An LLM agent proposes a change,
runs your benchmark, keeps the change if score improved past the noise
floor, reverts otherwise. Repeats until told to stop. Adapted from
Karpathy's autoresearch pattern; the only domain-specific bit is that the
metric is "kernel/e2e throughput" instead of "training loss."

Three files:

| file | what it is |
|---|---|
| `bench.py` | The verdict tool. Runs your bench N times, checks correctness, compares to baseline, emits KEEP/REVERT JSON. |
| `prompt.md` | The agent prompt. Describes the loop. Hand this to your LLM agent. |
| `README.md` | This file. |

That's the whole product. No orchestration, no sandbox, no daemon, no
manifest of hypothesis classes.

## Use

```bash
# 1. Tell bench.py how to run your bench (must print "score: <number>" to stdout)
export AUTOKERNEL_BENCH_CMD="python my_bench.py --some-flags"
export AUTOKERNEL_TEST_CMD="pytest tests/"   # optional; "" to skip
export AUTOKERNEL_TRIALS=3                   # default; raise if your bench is noisy

# 2. Establish the baseline (first call writes baseline.json with no comparison)
python bench.py

# 3. Start your agent with the prompt
#    (whatever your agent CLI is — Claude Code, Cursor, Aider, etc.)
your-agent < prompt.md
```

The agent will read `prompt.md`, run the loop, and write each accepted
change as a git commit. `SESSION_LOG.md` is its memory between attempts.

## Configuration

All flags optional; sensible defaults baked into `bench.py`:

| env var | default | what |
|---|---|---|
| `AUTOKERNEL_BENCH_CMD` | (required) | shell command, must print `score: <float>` |
| `AUTOKERNEL_TEST_CMD` | `""` | correctness command; exit 0 = pass; empty skips |
| `AUTOKERNEL_TRIALS` | `3` | how many bench runs to average |
| `AUTOKERNEL_TIMEOUT_S` | `600` | per-trial timeout |
| `AUTOKERNEL_SIGMA_K` | `2.0` | improvement must exceed K * combined_sigma |

## Why this is short

Earlier versions of this repo had ~1600 lines of harness, sandbox, watchdog,
classifier, multi-GPU orchestrator, and a five-way verdict taxonomy. None of
that was the value prop. The value prop is the loop: *propose, measure,
keep or revert, repeat.* Karpathy figured this out for LLM training; we are
applying it to GPU kernels. Everything else was scope creep added in
response to specific incidents — and most incidents are better surfaced to
the human than papered over with more code.

If you need 8-GPU multi-campaign orchestration, run 8 instances of this in
8 terminals. The "global loss log" is `git log --all --oneline` across
branches.

## License

See `LICENSE`.
