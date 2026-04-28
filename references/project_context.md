# Project context

> **Template note.** This file is the *Genesis-on-MI300X instance* of the
> project context template. To retarget the harness at a different project
> (e.g. Composable Kernel benchmarks, AITER, hipBLASLt), replace the contents
> below with your project's equivalents, keeping the section structure
> (E2E baselines / Top-N kernel gap / Layer model / Shipped
> patterns / Hard-coded gotchas). The harness reads this file by path; nothing
> in `bench.py` or the agent loop hard-codes the project name.

# the production workload / Genesis-on-MI300X — Context (FROZEN)

This file is the agent's source of truth for what's already known about this
specific project. Numbers here are baselines and targets — do not change them;
the orchestrator reads them to compute "improvement" deltas. Last updated from
`Genesis_Benchmark(April_21).csv` and `h100_vs_mi300x_kernel_comparison.txt`.

## Workload

the production workload team. Workload: `benchmark_scaling.py --precision 32 --max-envs 8192 --num-steps 500`. Unitree G1 29-DoF robot, CG solver, 15 iterations, 1e-6 tolerance.

## E2E baselines

| Run | Wall (s) | Throughput (env·steps/s) | % of H100 |
|---|---|---|---|
| H100 | 5.157 | 794,280 | 100.00% |
| MI300X baseline (pre-Apr-15) | 17.089 | 239,688 | 30.18% |
| MI300X current (Apr 21) | 9.456 | 433,156 | 54.54% |
| MI300X target (Apr 30) | 6.445 | 615,000 | **80.00%** |
| MI300X stretch (parity) | 5.157 | 794,280 | 100.00% |

**The target this autokernels run is chasing: 80% of H100 by 30 Apr 2026.**

## Top-8 kernel gap (8192 envs / 500 steps / FP32)

From `h100_vs_mi300x_kernel_comparison.txt`, ranked by absolute MI300X total time:

| # | Kernel | MI300X (ms) | H100 (ms) | Ratio | % of E2E gap | Status (Apr 21) | Owner |
|---|---|---|---|---|---|---|---|
| 1 | `func_solve_body_monolith_c484_kernel_1` | 6,400 (post-fix) | 3,520 | 3.38× | 56.5% | DONE (compiler patch — AccVGPR + global_load) | (redacted) |
| 2 | `kernel_step_1_c498_kernel_10_factor_mass` | 192.35 (post-fix) | 145.1 | 1.33× | 4.9% | DSL retune landed (BLOCK_DIM=64, wave64) | (redacted) |
| 3 | `_func_narrowphase_multicontact_mixed` | 230 (current) | 183.8 | 1.25× | 3.9% | Investigating | (redacted) |
| 4 | `_func_narrowphase_contact0` | 361 (current) | 219.8 | 1.65× | 2.1% | Investigating | (redacted) |
| 5 | `kernel_step_2_c500_kernel_9_update_cartesian_space` | 742 (current) | 158.6 | 4.68× | 4.6% | Investigating (perf PR in flight) | (redacted) |
| 6 | `kernel_step_1_c498_kernel_5_mass_mat_assemble` | 464.7 | 269.7 | 1.72× | 2.1% | Investigating | (redacted) |
| 7 | `func_broad_phase_c400_kernel_{0,1,2}` | 449.61 (current) | 35 | 12.85× | 7.1% | In progress (TICKET, eliminating scalar ops + bitfields) | (redacted) |
| 8 | `func_solve_init` | 424.30 | 76.01 | 5.58× | — | — | (redacted) |

## What this autokernels run targets

Two campaigns, named to match the user's terse spec ("`kernel_step_1_cX_X_kernel_11_func_broad_phase_ETC`"):

### Campaign A — `func_broad_phase`

- **Per-kernel ratio**: ~12.85× MI300X/H100 — the worst relative slowdown in the top 8.
- **Source**: `genesis/engine/solvers/rigid/collider/broadphase.py` (~396 lines).
- **Tracker**: TICKET, owner (redacted). Strategy noted: "eliminating scalar ops and using bitfields where possible."
- **Closing gap to 1.5× would save**: ~410 ms / step over 100 steps → ~2.0% E2E uplift.

### Campaign B — `kernel_step_1_2`

- The **kernel_step_1** and **kernel_step_2** family of `qd.kernel`s in `genesis/engine/solvers/rigid/rigid_solver.py:2559-2622` collectively cover Step1-Kernel-{5,10,11,25} and Step2-Kernel-{1,9,10}.
- **Top sub-kernel currently un-optimized**: `kernel_step_2_c500_kernel_9_update_cartesian_space` (4.68× MI300X/H100, 4.6% of E2E gap).
- **Factor_mass already shipped** (reference patch): 71% kernel reduction from BLOCK_DIM=64 + wave64 retune. **This is the proof-of-concept** that DSL retunes on this family work.
- **Closing the cartesian_space + mass_mat_assemble gaps to 1.5×** would be ~9-12% E2E uplift on top of current.

## Layer model (which knob to turn)

Three layers, all in repos already on disk:

1. **DSL** — `$HOME/work/Genesis/` Python. Edits: `block_dim` values, wave64 patterns, `qd.simt.warp.shfl_down_*` (only enabled on CUDA — check before relying on AMD path), `qd.func` inlining, scalar-op elimination. **Cycle: 30-90s untraced, 2-3 min traced.** Cache invalidation required (`rm -rf /root/.cache/quadrants /root/.cache/mesa_shader_cache`). This is the campaign's primary action surface.
2. **Compiler** — `$HOME/work/quadrants/` C++. Edits: `runtime/amdgpu/jit_amdgpu.cpp:83` (`amdgpu-waves-per-eu`), `codegen/amdgpu/codegen_amdgpu.cpp:160-193` (`optimized_reduction()` FP64), `codegen/amdgpu/codegen_amdgpu.cpp:429-442` (block_dim 32→64 silent promotion). **Cycle: 30-90 min Quadrants rebuild.** Kept out of the per-experiment loop; the agent may propose compiler patches but the human runs them out-of-band.
3. **LLVM/runtime** — owned by AMD compilers team via LCOMPILER-1748. Out of scope here.

The agent's loop is **DSL only by default**. If a DSL ratchet plateaus and the agent identifies a compiler-layer hypothesis, it is logged to `workspace/compiler_proposals.md` for human review — not executed in the loop.

## Already-shipped patterns the agent should know

From the tracker:

- **`factor_mass`** (reference patch): explicit `BLOCK_DIM=64`, `WARP_SIZE=64`, `# wave64: all threads in lockstep` comment. 671 ms → 192 ms (71% kernel reduction).
- **`solve_body_monolith`** (reference patch): compiler-layer patch — AccVGPR allocation + change `flat_load` to `global_load` via codegen optimization pass. 9.83% E2E uplift.
- **`add_inequality_constraints`** (reference patch): chain-walk dedup across friction-pyramid rows, inlined identity-quat simplification of `qd_transform_motion_by_trans_quat`, batched `atomic_add(counter 4)` per contact, explicit `block_dim=64`. kernel_3 −29.5%, kernel_5 −10.7%, total add_inequality −23.5%, E2E +2.71% (3-trial median 297k → 305k env·steps/s, ~7σ above noise). Net +12 LoC. **This is the canonical reference pattern for what a DSL win looks like**.

The agent should read those patches in `git log` on the relevant Genesis branches if it wants concrete examples.

## Hard-coded gotchas the harness handles for you

- `GS_FAST_MATH=0` — set in `bench.py`'s container env. Don't unset.
- Kernel cache wipe — `bench.py` runs `rm -rf /root/.cache/quadrants /root/.cache/mesa_shader_cache` before every benchmark.
- `runtime_initialize_rand_states_serial` — one-shot init, ≥50% of profile time. **`bench.py` strips it** from the per-step attribution.
- rocprofv3 ~30% overhead — `bench.py` runs untraced for headline throughput, then a separate (shorter, --num-steps 100) traced run for kernel attribution.
- `HIP_VISIBLE_DEVICES=N` — set per container by the launcher. One container = one GPU.
