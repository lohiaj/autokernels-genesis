# Contributing

Issues and pull requests are welcome — both for the simple flow (`program.md` + `sandbox.py`) and the advanced 8-GPU mode (`program-advanced.md` + harness Python).

## Filing an issue

The most useful issues include:

- The **kernel name** you were optimizing.
- The **repo + commit** of Genesis (and/or Quadrants) the sandbox cloned (`uv run sandbox.py status`).
- The **first message** you sent the agent.
- The **last 50 lines** of the agent's output (the bench run, the rubric commit message, what the agent decided).
- The contents of `results.tsv` and `learning.md` if they exist.

Bug labels we care about most:
- `recon-stuck` — agent couldn't find the source/bench/test for a kernel name.
- `false-keep` — agent kept a regression that 3+ trials would have caught.
- `false-revert` — agent reverted a real win because the noise floor was too high.
- `sandbox-leak` — agent edited or git-reset something OUTSIDE `~/.cache/autokernels-genesis/sandbox/`. (This is a hard bug — please file with the exact path that got touched.)

## Sending a PR

Small PRs land fast. Things we'd love:

- **More per-kernel `classes.json` entries** for the advanced mode. If you maintain a kernel family, write a `kernels/<your_campaign>/classes.json` with the regex vocabulary that classifies your hypotheses. See `_classify.py::DEFAULT_PATTERNS` for the format.
- **Sandbox URL defaults** for AMD-internal repos. If your team has a stable URL for Quadrants (or another tool) and is willing to make it the public default, send a PR that updates `sandbox.py::REPOS`. Keep the env-var override path so other teams can point elsewhere.
- **Better recon heuristics in `program.md`**. If you have a kernel where the agent stalled at A1/A2/A3 because the find/grep pattern missed it, document the pattern that *would* have worked and add it to the `program.md` reconnaissance phase.
- **Profiler integrations beyond `rocprofv3` / `omniperf`**. If you'd like the harness to read `rocprof-compute` or vendor-specific profiler output, see how `bench.py::parse_kernel_stats` does it for `rocprofv3` and follow the same pattern.

## What NOT to PR (without discussion)

- Don't relax the B1.5 rubric, the 2σ KEEP rule, or the never-stop semantics. They're load-bearing — every relaxation we tried while building this caused the loop to plateau at 4-5 KEEPs. If you have evidence one of them is wrong, open an issue first with data.
- Don't add prerequisites (Docker, a server, a service). The 4-step quickstart is the product.
- Don't add per-vendor lock-in. The harness should stay portable across ROCm projects (Genesis, Quadrants, Composable Kernel, AITER, hipBLASLt, custom). If your change only makes sense for one project, put it in that project's per-campaign manifest, not the harness.

## Local checks before pushing

Quick sanity sweep:

```bash
# Python syntax
python3 -c "import ast; [ast.parse(open(f).read()) for f in [
    '_classify.py','_config.py','bench.py','orchestrate.py','prepare.py',
    'sandbox.py','summarize.py','watchdog.py','global_log.py'
]]; print('OK')"

# Bash syntax
bash -n launcher/launch_8gpu.sh

# Sandbox smoke (clones a tiny test repo, doesn't touch real Genesis)
AUTOKERNEL_SANDBOX=/tmp/ak-test \
  AUTOKERNEL_GENESIS_URL=https://github.com/octocat/Hello-World.git \
  AUTOKERNEL_GENESIS_BRANCH=master \
  uv run sandbox.py setup --kernel ci_smoke
uv run sandbox.py wipe --yes
```

## Commits

- Use a short, lowercase imperative subject (e.g. `sandbox: refresh on existing clone instead of wipe`).
- The body explains *why* the change matters, not what the diff already shows.
- No need for sign-off, CLA, or attribution footers.
