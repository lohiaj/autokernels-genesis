# `kernel_step_1_2` playbook

This is the agent's seed list of ideas for the `kernel_step_1` / `kernel_step_2` family.

Source files in scope:
- `genesis/engine/solvers/rigid/rigid_solver.py` (kernel_step_1 at ~L2559, kernel_step_2 at ~L2622)
- `genesis/engine/solvers/rigid/abd/forward_dynamics.py` (canonical reference: func_factor_mass at L574)

The Step1/Step2 family is **6 kernels** in the rocprofv3 top-25 — this is the highest-leverage campaign measured in cumulative MI300X time (~3,400 ms across the family vs ~1,000 ms for H100).

Sub-kernel current ratios (from `h100_vs_mi300x_kernel_comparison.txt`):
- `kernel_step_1_c498_kernel_10_factor_mass`: 7.22× (DSL-retune landed → 1.33×)
- `kernel_step_1_c498_kernel_5_mass_mat_assemble`: 1.93×
- `kernel_step_1_c498_kernel_25`: 2.82×
- `kernel_step_1_c498_kernel_11`: not directly named in comparison; likely related to `func_solve_init_kernel_11_update_gradient_tiled` (3.66×)
- `kernel_step_2_c500_kernel_9_update_cartesian_space`: 3.01×
- `kernel_step_2_c500_kernel_1`: 2.40×
- `kernel_step_2_c500_kernel_10_forward_velocity_entity`: 2.33×
- `kernel_step_2_c500_kernel_7`: 5.42×

The **proven reference pattern** is `factor_mass`: BLOCK_DIM=64, WARP_SIZE=64, explicit `# wave64: all threads in lockstep`. Read the diff in `git log` on `$HOME/work/Genesis` (`TICKET`) before making the first edit. **That diff is your model.**

---

## Tier 1 — apply factor_mass-style retune to siblings

The same pattern almost certainly applies to `mass_mat_assemble` (kernel_5), `update_cartesian_space` (step_2_kernel_9), and `forward_velocity_entity` (step_2_kernel_10) since they're structurally similar (per-batch, per-link state updates with inner reductions).

Action:
1. Read `func_factor_mass` in `abd/forward_dynamics.py:574`. Note: explicit `BLOCK_DIM = 64`, the `WARP_SIZE = 64` constant, the wave64 comment, the loop_config.
2. Find the analogous `qd.func` or inner block in `rigid_solver.py` for `kernel_5` and `step_2_kernel_9`. Apply the same three changes.
3. Bench. Expected: 30-60% per-sub-kernel reduction on the first attempt; this is the lowest-hanging fruit in the entire codebase right now.

## Tier 2 — kernel_step_1 inner loops: block_dim audit

Inside the giant `kernel_step_1` (`@qd.kernel` at rigid_solver.py:2559), grep for nested `qd.loop_config(block_dim=...)`. Each of those becomes a numbered sub-kernel in the rocprofv3 output.

Action: for each `block_dim=32` site, change to explicit `block_dim=64`. For sites where the body has no cross-thread dependencies in the outer dim (per-batch loops), try `block_dim=128`. Re-bench between each one — don't combine.

## Tier 3 — fuse adjacent passes

Some Step1 sub-kernels read state, transform it, write it back, then the next sub-kernel reads what the previous one wrote. That's two HBM round-trips for what could be one.

Action: identify pairs of adjacent `qd.func`/inline blocks in `kernel_step_1` that share intermediate state. Inline the second into the first; keep the intermediate in registers/LDS rather than HBM.

Caveat: this changes the rocprofv3 sub-kernel breakdown — the new fused kernel won't match `kernel_5_mass_mat_assemble_*` regex anymore. Update the per-experiment expectation accordingly (the per-kernel attribution might briefly look "worse" because it's combined; the e2e number is what matters then).

## Tier 4 — reduce via warp shuffle (CONDITIONAL)

If `quadrants/codegen/amdgpu/codegen_amdgpu.cpp` exposes a working `shfl_down_f32` for AMD wave64, use it for inner reductions in mass_mat_assemble and update_cartesian_space (currently O(BLOCK_DIM) serial per `solver.py:1955`).

Verify FIRST:
```bash
grep -n "shfl_down\|wave_shfl\|ds_swizzle" /src/quadrants/quadrants/codegen/amdgpu/codegen_amdgpu.cpp
```

If the function exists and is wired through `qd.simt.warp`, generalize the `ENABLE_WARP_REDUCTION` gate (`solver.py:1904`) to `(_IS_NV or _IS_AMD)`. If it doesn't exist, **don't** — it's a compiler-layer item, log to `workspace/compiler_proposals.md`.

## Tier 5 — Cartesian update specifically

`kernel_step_2_kernel_9_update_cartesian_space` is at 4.68× MI300X/H100 with 4.6% of the e2e gap. There's a perf PR in flight (reference patch — check `git log` on `ROCm/Genesis` perf branches for `revert-5-revert-4-vrachuma/perf/improve_cartesian_update`).

Action:
1. `git log --oneline | grep -i cartesian` on `$HOME/work/Genesis`. If the PR's commits are there, read the diff for inspiration.
2. The kernel transforms per-link state from joint space to Cartesian. Look for chains of `qd.transform` calls where the rotation can be hoisted out of the per-batch loop.

## Tier 6 — LDS staging for large per-batch state

The Step1 sub-kernels often touch the entire 29-DoF state (mass matrix is 29×29 = 841 floats per env = 3.3 KB per env in fp32). At 8192 envs that's >25 MB — well above LDS, so it's HBM-resident. But within a workgroup, you can stage one env's 3.3 KB into LDS (well below 64 KiB/CU).

Action: in the per-batch inner loop, stage the relevant slice into LDS once at workgroup start, do all the per-batch work against LDS, write back at the end. Watch for LDS bank conflicts — pad row stride if you see slowdown.

This is complex but high-leverage if you nail it.

---

## Anti-patterns specific to this campaign

- **Don't optimize sub-kernels in isolation when they share state.** A 2× win on `kernel_5` that doubles the data the next sub-kernel must re-read is a wash or a regression on e2e.
- **Don't restructure `kernel_step_1`'s overall control flow.** It's a fixed pipeline (mass-assemble → factor → solve init → cg solve → ...). Re-ordering breaks correctness.
- **Don't touch the CG solver iteration count or tolerance.** That's a numerical-quality change, not a perf one. The team's production-grade baseline assumes `iterations=15, tolerance=1e-6`.
- **Don't add `print` or `qd.static_print` for debugging in shipped commits.** They serialize.

## What "good" looks like for this campaign

`factor_mass` shipped at -71% kernel time. If we land Tier 1 on three siblings (`mass_mat_assemble`, `update_cartesian_space`, `forward_velocity_entity`) at -30% each, that's roughly **5-10% E2E uplift**, on top of whatever Tier 2/3 yields. End-of-campaign, the family's contribution to the H100 gap drops from 18% to <8%.
