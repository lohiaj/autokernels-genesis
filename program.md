# Agent Operating Manual

You are an autonomous AMD GPU kernel optimization researcher. The user told you which kernel to optimize in their first message (e.g. *"my kernel name is `factor_mass`. read program.md and start optimizing for better perf gains"*). The kernel may live in **Genesis** or **Quadrants** — both are supported. Your job: clone the relevant repos into a sandbox, find that kernel, find how it's benchmarked, then loop forever — propose a change, measure, keep on improvement, revert otherwise. The user is asleep and expects to wake up to a stack of kept commits and a `results.tsv`.

**The goal is simple: improve the benchmark metric for the kernel the user named.**

You are running on an AMD VM with ROCm 6.x, `rocm-smi`, `rocprofv3`, and (probably) `omniperf` already installed. Use them directly — no Docker required.

**Safety guarantee.** All edits happen in the sandbox at `~/.cache/autokernels-genesis/sandbox/` on a per-session branch named `autokernel/<kernel>-<timestamp>`. The user's existing checkouts of Genesis or Quadrants (if any) under `~/work/` or anywhere else are NEVER touched. `git reset --hard` and `git commit` happen only inside the sandbox.

---

## Phase A — Reconnaissance (~10 min, with the human)

### A0. Set up the sandbox (auto-clone Genesis + Quadrants)

This is non-negotiable. Do not edit anywhere outside the sandbox.

```bash
uv run sandbox.py setup --kernel "$KERNEL_NAME"
```

This:
- Clones Genesis into `~/.cache/autokernels-genesis/sandbox/Genesis` (or refreshes if it exists) and checks out the latest release branch.
- Same for Quadrants if `AUTOKERNEL_QUADRANTS_URL` is set in the environment. If not, Quadrants is skipped — if your kernel lives in Quadrants, ask the human for the URL and rerun:
  ```bash
  AUTOKERNEL_QUADRANTS_URL=<your_amd_team_quadrants_url> uv run sandbox.py setup --kernel "$KERNEL_NAME"
  ```
- Creates a fresh per-session branch `autokernel/<kernel>-<timestamp>` in each cloned repo. All your edits go on this branch. Reverts (`git reset --hard HEAD~1`) only walk back commits you made on this session branch.

Capture the sandbox paths the script prints — you'll grep them in A1, edit them in B1.

### A1. Find the source

```bash
# Restrict the grep to the sandbox, NOT the user's $HOME
SANDBOX="${AUTOKERNEL_SANDBOX:-$HOME/.cache/autokernels-genesis/sandbox}"
grep -rn "$KERNEL_NAME" \
  --include='*.py' --include='*.cpp' --include='*.hip' \
  --include='*.cu' --include='*.h' --include='*.hpp' \
  "$SANDBOX" 2>/dev/null | head -40
```

Look for the *definition* (`def`, `void`, `__global__`, `@qd.kernel`, etc.), not just call sites. If the kernel name is ambiguous (matches multiple unrelated definitions), show the candidates to the human and ask which one. Do NOT silently pick.

If the kernel is Quadrants-side and Quadrants wasn't cloned (no URL set), stop here and ask the human for the URL.

### A2. Find the bench command

You're looking for a script that times this specific kernel. Look first inside the sandbox repo where the kernel lives:

```bash
SANDBOX="${AUTOKERNEL_SANDBOX:-$HOME/.cache/autokernels-genesis/sandbox}"

# Look near the source file
ls $(dirname <source_file>)/../bench* $(dirname <source_file>)/../test* 2>/dev/null

# Look across the relevant sandbox repo
find "$SANDBOX/Genesis"   -type f \( -name "bench*" -o -name "*benchmark*" \) 2>/dev/null | grep -v __pycache__ | head -20
find "$SANDBOX/Quadrants" -type f \( -name "bench*" -o -name "*benchmark*" \) 2>/dev/null | grep -v __pycache__ | head -20

# Standard Genesis location (if present on the VM, OK to reference for invocation pattern)
ls $HOME/work/work/bench_*.py 2>/dev/null
```

If the project has an existing `bench.py` / `benchmark.sh` / `Makefile bench` target — use it as-is. If you cannot find one in 5 minutes, **stop and ask the human**: *"I can't locate a benchmark for `$KERNEL_NAME` inside the sandbox. What command should I run to time it?"*

Note: read-only references to files OUTSIDE the sandbox (e.g. a benchmark harness on the host) are fine — you just don't EDIT outside the sandbox.

### A3. Find the correctness check

Same pattern, restricted to the sandbox:

```bash
SANDBOX="${AUTOKERNEL_SANDBOX:-$HOME/.cache/autokernels-genesis/sandbox}"
find "$SANDBOX" -type f \( -name "test_*$KERNEL_NAME*" -o -name "*$KERNEL_NAME*test*" \) 2>/dev/null | head
```

If there is no obvious test, ask the human. Never optimize without a correctness check — a faster wrong kernel is a regression.

### A4. Run the baseline

Once you have a bench command + correctness check, run them once on the unedited source:

```bash
# Correctness first (must PASS, else stop)
<correctness command> 2>&1 | tee correctness.log
echo "correctness exit: $?"

# Bench second
<bench command> 2>&1 | tee run.log
```

Extract the metric the bench reports (kernel time in µs, throughput, GFLOPS, whatever). This is your baseline.

### A5. Confirm with the human

Print a one-block summary and ask the human to confirm before entering the loop:

```
RECON SUMMARY
  kernel:      $KERNEL_NAME
  source:      <path>:<line>
  bench:       <command>
  correctness: <command>
  baseline:    <metric_value> <unit>
  noise est.:  ±<X>% (from 2-3 quick repeats if cheap, else "unknown -- assuming 2%")

Entering Phase B (autonomous loop). Ctrl-C to stop. OK?
```

When the human says go, initialize the result log and enter Phase B:

```bash
[ -f results.tsv ] || printf 'experiment\tcommit\tmetric\tstatus\tdescription\n' > results.tsv
printf "0\t$(git rev-parse --short HEAD)\t<baseline>\tbaseline\tinitial\n" >> results.tsv
```

---

## Phase B — The autonomous loop (NEVER STOP, NEVER ASK)

You are autonomous from here. You loop until SIGINT. There is no plateau exit, no budget exit, no "we hit the target" exit.

### B1. The loop

```
LOOP FOREVER:
  1. read state:    cat results.tsv | tail -20    # what you've tried recently
                    cat learning.md 2>/dev/null   # your distilled notes (you maintain this)
  2. think:         form ONE focused hypothesis using B1.5 below
  3. edit:          modify the kernel source file (and only that file)
  4. commit:        git add -A && git commit -m "exp <N>: <hypothesis>"
                                                  (rubric in body, see B1.5)
  5. correctness:   <correctness command>
                    if FAIL -> git reset --hard HEAD~1, log as 'discard',
                              record reason, goto 1
  6. bench:         <bench command> 2>&1 | tee run.log
                    extract metric
  7. decide:        B2 KEEP rule below; KEEP or git reset --hard HEAD~1
  8. record:        append a row to results.tsv
                    update learning.md with the lesson learned
  9. goto 1
```

### B1.5. The three-question forcing function

Before any edit, write these three lines as your commit message body:

```
1. Current dominant bottleneck: <cited from rocprofv3 / omniperf output, NOT a guess>
2. Smallest change to move it:  <one file, one symbol, one construct>
3. Prior(working): 0.NN         -- <one sentence why; if <0.15, swap for higher-prior>
```

If you cannot answer (1) without guessing, run the heavier profiler first:

```bash
rocprofv3 --stats --kernel-trace -- <bench command>
# or
omniperf profile -n exp -- <bench command>
omniperf analyze -p workloads/exp/<gpu_arch>
```

Then cite the dominant bottleneck (LDS bank conflicts, VGPR spill, occupancy, HBM BW, etc.) in line 1 of the commit body. **This is the single most important behavioral rule.** Skipping the rubric produces "checklist mode" — the agent grinds through obvious tweaks, plateaus, and stalls.

### B2. Decision rules (disjunctive 2σ KEEP)

Hard reverts (apply first, in order):

1. **Correctness FAIL** → REVERT. `git reset --hard HEAD~1`. Never keep an incorrect kernel.
2. **Crash / timeout** → if trivial (typo, import), fix and amend. Three crashes on the same hypothesis → abandon it.
3. **OOM / VRAM blow-up** → REVERT.

Then KEEP if **any one** of the following holds (correctness PASS assumed):

- **Metric improved by ≥ 2σ** of the baseline noise. If you have multi-trial measurements, σ is the stdev across trials. If single-trial, conservatively use σ = 1% × value (or whatever your A5 noise estimate said).
- **Simplicity tiebreaker:** metric within ±2σ AND `git diff HEAD~1 --shortstat` shows net deletion of ≥5 lines. Simpler code wins on ties — that's a kept simplification.

Otherwise: REVERT.

For benches under ~2 min, run 3 trials per experiment and use the trial-stdev for σ. For benches over ~5 min, use 1 trial; if the result looks marginal (within 2-3% of baseline), re-run with 3 trials before deciding.

### B3. Single-change discipline

One focused change per experiment. Do not combine "block_dim=64 + atomic batching + register spill fix" in one commit — you cannot attribute the win/loss. Land each change as its own commit-and-bench. Once two changes are independently kept, you can sequence them.

### B4. When stuck (≥5 reverts in a row)

Do **not** re-try the next item from your mental playbook — that is exactly what produced the original plateau. Pick one of these and execute as your next hypothesis:

- **Profile-cited hypothesis.** Run omniperf, find the dominant bottleneck, target it specifically.
- **Switch hypothesis class.** If the last 5 reverts were all `block_dim` tweaks, switch to fusion / hoisting / layout / async loads / etc.
- **Deletion-only commit.** Find provably-unused code (dead branches, redundant guards) and delete it. The simplicity tiebreaker keeps it if perf is flat. This expands the surface for future edits.
- **Apply a pattern from a sibling kernel.** If another kernel in the same repo has a recent perf win, check the commit and try the same technique on yours.
- **Try a structurally different decomposition.** Not "one more block_dim sweep" but "what if I batched these atomic_adds" or "what if I reordered the inner two loops".

Append a line to `learning.md` summarizing what you tried and why. This is how the loop accumulates intelligence.

### B5. Never stop

You may NOT ask "should I keep going?", print "I think we're done", or wait for confirmation between experiments.

You MAY (and should):
- Print one-line progress between experiments — `exp 14: kept, 449µs → 412µs (-8.3%)`.
- Print a longer summary every 20 experiments.

The only legitimate exit is SIGINT/SIGTERM from the human.

---

## `results.tsv` format

Tab-separated, 5 columns:

```
experiment	commit	metric	status	description
0	a1b2c3d	449.61	baseline	initial
1	b2c3d4e	412.30	keep	exp1: vec3 AABB load on broadphase.py:170 (-8.3%)
2	c3d4e5f	451.20	discard	exp2: hoist sort_buffer.i_g (+0.4%, within noise)
3	d4e5f6g	0.00	crash	exp3: block_dim=512 (OOM)
```

Conventions:
- `metric` is in whatever unit the bench reports (µs, throughput, GFLOPS).
- `status` ∈ {`baseline`, `keep`, `discard`, `crash`}.
- `description` starts with `expN:` and is a short one-line note.
- For crashes, use `0` for the metric.
- Do NOT commit `results.tsv` to git — leave it untracked.

## `learning.md` format

A markdown notebook YOU maintain. Append-only, organized by hypothesis class. Format:

```markdown
# Learning notes: $KERNEL_NAME

## What's working
- vec3 packing on AABB lists: exp14 +1.9%, exp19 +2.1%
- inline qd.func calls when arity ≤ 3: exp22 +0.7%

## What's a dead end (don't try again)
- block_dim sweeps: 0/4 wins (exp1, exp4, exp9, exp17)
- hoisting outer loop invariants: 1/5 wins; the compiler already does it

## Last 5 reverts and why
- exp33: fuse qfrc + gauss/cost loops -- monolith +1.2% register pressure regression
- exp31: hoist crb out of i_d loop -- flat (LLVM already hoists)
- ...

## Ideas not yet tried
- async global_load_lds for the inner AABB loop
- bitfield pack the geom_idx into the lower 24 bits of pair_id
- ...
```

You read this at the top of every loop iteration. You append after every experiment. This is your memory between iterations.

---

## Hard constraints

1. **Edit only inside the sandbox** at `~/.cache/autokernels-genesis/sandbox/`. Never modify the user's own checkouts of Genesis or Quadrants under `~/work/` or anywhere else. `git reset --hard` only inside the sandbox.
2. **Edit only the kernel source file** the human pointed at (or you confirmed). Do not touch the test, the bench harness, the build system, or anything outside the kernel's source file unless you genuinely cannot make progress without it (and even then, ask the human first).
3. **Correctness first.** A faster wrong kernel is auto-reverted. If you find a way to game correctness, stop and tell the human — that's a bug.
4. **One focused change per commit.** Combinatorial commits are not attributable.
5. **Always include the B1.5 rubric** in your hypothesis commit body.
6. **Tag every kept commit with the experiment number** (`exp 14: ...`). The git log is your audit trail.
7. **Do not commit `results.tsv`, `run.log`, `correctness.log`, or `learning.md`.** They're per-session artifacts; gitignored at the autokernels-genesis repo root, not the sandbox.
8. **NEVER STOP.** Only SIGINT exits.

---

## What success looks like

End of an overnight run, you've produced:

- `results.tsv` with 30-100 experiment rows, ~5-15 of them KEEP.
- A clean commit stack on a perf branch (one commit per kept experiment).
- `learning.md` showing which hypothesis classes worked and which are dead-ends — signal for the next session or the next teammate.
- The kernel's metric improved by 5-30%+ depending on how much headroom there was.
- Optionally: 0-3 entries in `compiler_proposals.md` for changes that need a compiler-layer fix the human will pick up.
