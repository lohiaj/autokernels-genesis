# `func_broad_phase` playbook

This is the agent's seed list of ideas for this campaign. Order them by leverage (top first), but don't restrict yourself — the playbook is a starting point, not a plan.

Source files in scope:
- `genesis/engine/solvers/rigid/collider/broadphase.py` (~396 lines)
- `genesis/engine/solvers/rigid/collider/utils.py` (shared helpers)

Currently 12.85× MI300X/H100. The kernel is **memory-bound** in profile data — most wins will come from access patterns + occupancy, not arithmetic.

---

## Tier 1 — explicit `block_dim=64` (the wave64 default)

Quadrants AMD codegen silently promotes `block_dim=32 → 64` for correctness (`codegen_amdgpu.cpp:429-442`) but does NOT rethink shared-mem tile sizes. Result: same shared-mem footprint, 2× the threads contending, default sub-optimal occupancy.

Action: grep `block_dim=` in `broadphase.py` and `utils.py`. For each call site:
- If `block_dim=32` → change to **explicit `block_dim=64`** as the first move (matches what the silent promotion does, but lets the compiler reason correctly about it).
- If the kernel is outer-loop-parallel (per-batch, no cross-thread dep), try `block_dim=128` or `256` next. Bigger wave packing → fewer kernel launches → better latency hiding.
- Symptom of going too large: register spilling (visible in AMDGCN as `scratch_load`/`scratch_store`), kernel slowdown.

Reference patch that worked: `factor_mass` ((redacted), TICKET) — 671 ms → 192 ms (71%) from `BLOCK_DIM=64, WARP_SIZE=64, # wave64: all threads in lockstep`.

## Tier 2 — eliminate scalar ops, use bitfields

Per the TICKET strategy comment from (redacted). Broad-phase pair generation often involves boolean masks (which AABB pairs overlap) — naively done with byte/int per-pair, wastefully wide.

Action: where the kernel writes per-pair flags or per-geom collision masks:
- Replace `int` flags with bitfield packs (`u32` carrying 32 booleans).
- Replace per-element `qd.atomic_add(counter, 1)` with batched accumulation (write to a thread-local register, atomic_add the batched count once at the end).
- Look for `qd.func` calls that return a tuple of small ints; consider packing them into a single `u32`/`u64`.

Reference patch that worked: `add_inequality_constraints` ((redacted) + jlohia, TICKET) — `batched atomic_add(counter 4) per contact` was one of the wins; net +12 LoC; -23.5% kernel; +2.71% E2E.

## Tier 3 — `qd.func` inlining & identity-quat simplification

Some `qd.func` calls do work that's logically a no-op when one input is identity (e.g., `qd_transform_motion_by_trans_quat` reduces to a couple of scalar ops when `quat == identity`). Broad-phase frequently transforms world-space AABBs through identity poses for static geoms.

Action: grep for `qd_transform_*`, `qd_quat_*`, `qd.quat_*` calls in broadphase.py and the immediate `qd.func` callees. For any whose inputs you can prove (statically or with a `qd.static(... == identity)` check) are identity, inline the simplified math.

Reference patch: `add_inequality_constraints`'s `inlined identity-quat simplification of qd_transform_motion_by_trans_quat (vel = cdot_vel - t_pos.cross(cdof_ang))`.

## Tier 4 — chain-walk dedup

Broad-phase often generates the same (geom_a, geom_b) pair from multiple traversal paths (e.g., friction-pyramid rows or BVH child overlap). Dedup at the producer is cheaper than dedup at the consumer.

Action: look for nested loops over geom_a × geom_b where the inner loop emits pairs. If two outer iterations can produce the same pair, hoist the dedup check to the outer scope or reorder the loop so duplicates are impossible.

Reference patch: `chain-walk dedup across 4 friction-pyramid rows` from TICKET.

## Tier 5 — AABB intersection structure

The current broad-phase is "all-vs-all" by name (`_func_broad_phase_all_vs_all_*`). At 8192 envs × N geoms per env, the pair count is quadratic. If the AABB layout in memory is per-env-major (likely), the inner loop is reading from far-apart memory locations.

Action:
- Check the layout of `geoms_state.aabb_min/max` in `array_class.GeomsState`. If it's `[i_geom, i_env, 3]`, the per-env loop strides poorly.
- A relayout is risky (touches many other kernels) — instead, add a **prefetch-style hoist**: load all of an env's AABBs into a thread-local `qd.static` array once, then loop, reusing.

This is a Tier 5 move because it's the most invasive. Try Tiers 1-4 first.

## Tier 6 — async global → LDS copies

CDNA3 supports `global_load_lds` (skip the register stage). For broad-phase's AABB sweep, this can hide global memory latency under compute.

Action: this is **compiler-layer**. The Quadrants DSL doesn't expose it directly. If you reach the point where Tier 1-5 is exhausted, write a `workspace/compiler_proposals.md` entry with a concrete code change to `quadrants/codegen/amdgpu/codegen_amdgpu.cpp` to emit `global_load_lds` for tiled-load patterns. **Don't try to do this in the loop**; rebuild cycle is 30-90 min.

---

## Anti-patterns specific to this campaign

- **Don't go below `block_dim=64`.** Compiler silently promotes anyway and the result is worse than picking 64 explicitly.
- **Don't add `qd.simt.warp.shfl_down_*` calls.** Per `solver.py:1894`, those are CUDA-only stubs; on AMD they fall through to serial reduction. Verify in `quadrants/codegen/amdgpu/codegen_amdgpu.cpp` before relying on any warp intrinsic.
- **Don't change AABB tolerances or collision thresholds.** That's a correctness change masquerading as a perf change. The pytest will catch it but only after a wasted cycle.
- **Don't chase `kernel_collision_clear_range_for`** in isolation. It has a 7.88× ratio but only 53.8 us total (rank 23). Get the bigger fish first.

## What "good" looks like for this campaign

A kept commit that reduces `kernel_avg_us` from ~450us toward ~100us (i.e., toward 3x H100, an intermediate target). One-shot 2x improvements via Tier 1 are realistic. Stacking Tier 2-4 wins to push toward 4-5x of current is the campaign goal. End-of-campaign: ~30-50% E2E uplift attributable to this kernel alone if you reach 5x.
