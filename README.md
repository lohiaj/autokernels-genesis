# autokernels-genesis

Autonomous LLM-driven research loop for AMD GPU kernel optimization. Hand it a kernel name; it loops forever — proposes a change, benchmarks, keeps on improvement, reverts otherwise — until you Ctrl-C.

## Quick start (4 steps)

```bash
git clone https://github.com/lohiaj/autokernels-genesis.git
cd autokernels-genesis
claude    # or `codex` -- any Claude Code / agent CLI
```

Then in the agent CLI, type:

```
my kernel name is YOUR_KERNEL_NAME. read program.md and start optimizing for better perf gains.
```

That's it. The agent reads `program.md`, locates your kernel in the source tree, finds its bench + correctness command, runs a baseline, confirms with you, then loops until you stop it.

## Prerequisites

The agent assumes you're on an AMD VM with the standard toolchain pre-installed:

- ROCm 6.x (`rocm-smi`, `rocprofv3`, `omniperf`)
- Python 3.10+
- Git
- Your kernel's source repo cloned somewhere under `$HOME`

No Docker, no setup script, no harness configuration. The agent does the discovery itself.

## What the agent does

1. **Reconnaissance** (~10 min, with you in the loop): `grep` for your kernel by name, find the bench/test commands, run the baseline. Confirms with you before going autonomous.
2. **Loop forever**:
   - Form one focused hypothesis (must answer the three-question rubric in the commit body — bottleneck, smallest change, prior probability).
   - Edit the kernel source file.
   - Run correctness; revert on FAIL.
   - Run the bench; KEEP if metric improved by ≥ 2σ, else revert.
   - Append to `results.tsv`; update `learning.md` with the lesson.
3. **You wake up** to 30-100 experiments, 5-15 of them kept on a clean commit stack, and a `learning.md` summarizing which idea classes worked.

Full details — the loop, the rubric, the KEEP rule, the stuck handler — live in [`program.md`](program.md). The agent reads it; you don't have to.

## Output

After an overnight run, `cd` into your kernel's source repo and:

```bash
git log --oneline                    # the kept commits, one per experiment
cat $HOME/autokernels-genesis/results.tsv   # full attempt log (kept + reverted + crashed)
cat $HOME/autokernels-genesis/learning.md   # what worked / what's dead-end
```

## FAQ

**Q: Does it only work for Genesis?**
A: No — it works on both **Genesis and Quadrants** kernels. The agent searches both sandbox repos for your kernel name and figures out which one to edit. Genesis is cloned by default; for Quadrants set `AUTOKERNEL_QUADRANTS_URL=<your_amd_team_url>` before running.

**Q: Will it corrupt my existing changes or upgraded kernels I'm working on?**
A: No. The program runs in a **git sandbox** at `~/.cache/autokernels-genesis/sandbox/`. Your existing checkouts of Genesis or Quadrants (under `~/work/` or anywhere else) are never touched. Each session creates a fresh branch named `autokernel/<kernel>-<timestamp>` so even repeated runs don't collide. `git reset --hard` only happens inside the sandbox.

**Q: Will I have to pull the Quadrants and Genesis repos into the dir?**
A: No. The program **pulls the latest release branch of the repos automatically** on first run via `sandbox.py setup`. Subsequent runs `git fetch + reset --hard` to keep up. Your `cwd` and your own clones are not modified.

**Q: Are contributions open?**
A: Yes — issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the (very short) guide.

## Advanced: 8-GPU multi-campaign mode

For coordinated runs across 8 MI300X with multiple kernel campaigns running in parallel (the original Genesis-on-MI300X integration), see [USAGE.md](USAGE.md). That mode adds:

- A campaign manifest system (`kernels/<campaign>/target.json`)
- A frozen bench harness (`bench.py`) with multi-trial sigma + GPU clock pinning + rubric enforcement
- A cross-agent shared loss log (`global_log.py`) so 8 agents avoid duplicate dead-ends
- A watchdog daemon (`watchdog.py`) for indefinite-run safety + drift detection
- An 8-GPU launcher (`launcher/launch_8gpu.sh`)

You don't need any of that for a single-kernel single-GPU optimization session — the simple flow above is the whole product.

## License

MIT — see [LICENSE](LICENSE).
