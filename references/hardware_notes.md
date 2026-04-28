# Hardware notes

> **Template note.** This file is the *MI300X / CDNA3 / gfx942 instance* of
> the hardware notes template. Replace contents with your GPU's chip details,
> peak compute table, common gotchas, and profiler invocations to retarget.
> Keep the section structure so the agent's expectations stay stable.

# MI300X / CDNA3 / gfx942 — Hardware cheatsheet (FROZEN)

Read once at setup. Do not modify mid-experiment.

## Chip

- **AMD Instinct MI300X**, CDNA3 architecture, ISA `gfx942`.
- 304 Compute Units (CUs), grouped as 8 XCDs (chiplets) × 38 CUs each.
- 4 SIMDs per CU. Each SIMD holds up to 10 in-flight wavefronts (so up to 40 waves/CU peak occupancy).
- **Wavefront size: 64 lanes.** This is the single most important number to internalize. CUDA-shaped code that assumes 32 silently breaks or runs at half-throughput.
- 192 GB HBM3, **5.3 TB/s peak bandwidth**.
- L2: 4 MiB per XCD (so 32 MiB total, but not unified — XCD locality matters).
- LDS (Local Data Share, = NVIDIA's shared memory): **64 KiB per CU**, **32 banks × 4 bytes wide**.
- Registers: 256 VGPR + 256 AGPR per SIMD-thread. AGPR = "accumulator VGPR", separate file for MFMA destinations.

## Peak compute (FP32 / 8192-env benchmark relevant)

- FP32 vector: ~163 TFLOPS.
- FP32 MFMA (matrix): ~163 TFLOPS (no FP32 matrix uplift on CDNA3 vs vector).
- FP16 MFMA: ~1,307 TFLOPS.
- BF16 MFMA: ~1,307 TFLOPS.
- FP64 vector: ~81 TFLOPS.
- FP64 MFMA: ~163 TFLOPS.

The production benchmark is FP32 per project requirements — MFMA is mostly not the lever; bandwidth and occupancy are.

## Concrete patterns the Genesis kernels keep tripping over

### 1. `block_dim=32` in `qd.loop_config(block_dim=32)`

CUDA-shaped (32-lane warp). Quadrants AMD codegen silently promotes to `block_dim=64` for correctness (`codegen_amdgpu.cpp:429-442`) but does NOT rethink shared-mem tile sizes. Result: same shared-mem footprint as if 32 threads, 2× the threads contending. **Action**: explicit `block_dim=64` (or 128/256 for outer-loop-parallel kernels). Verify by grep'ing `block_dim=32` and rewriting site-by-site.

### 2. `qd.simt.warp.shfl_down_f32(qd.u32(0xFFFFFFFF), ...)` is dead code on AMD

`solver.py:1894` admits this in a docstring. The CUDA fast path uses warp shuffle for reductions; on AMD it falls through to O(BLOCK_DIM) serial reduction. **Action**: only relevant if the AMD shuffle is wired up. Check `quadrants/codegen/amdgpu/codegen_amdgpu.cpp` for `shfl_down` before relying on it; otherwise, the only DSL-side win is bigger `block_dim` to reduce per-warp serial work.

### 3. `amdgpu-waves-per-eu=1,2`

Uniform setting in `quadrants/runtime/amdgpu/jit_amdgpu.cpp:83`. 2 waves/EU × 4 EUs/CU = 8 waves/CU = ~20% of CDNA3 peak occupancy. An ad-hoc allow-list at lines 70-80 exempts 5 CG-inner kernels by name substring. **Action (compiler-layer, out-of-loop)**: per-kernel attribute or remove uniform constraint. Memory-bound kernels (broad_phase, narrowphase, cartesian_update) typically gain disproportionately.

### 4. LDS bank conflicts (32 banks, 4 B)

Address pattern that hits the same bank from different lanes in a wave serializes. Stride-1 `float` accesses are conflict-free. Stride-32 `float` (every lane same bank) is 32-way serial. **Action**: pad arrays (`shared float a[N+1][M]` not `[N][M]`) or rewrite indexing to spread across banks. Symptom in profiler: low LDS bandwidth despite the kernel being LDS-bound.

### 5. `flat_load` vs `global_load`

Quadrants codegen manually inserts `addrspace(1)` casts on every load/store (`codegen_amdgpu.cpp:270-391`) so LLVM emits `global_load` instead of the slower `flat_load`. **Action**: when reading the AMDGCN dump, look for `global_load_dwordx2` (good) vs `flat_load_dwordx2` (bad). If you see `flat_load`, the addrspace-cast pattern was missed; report to compiler team.

### 6. `optimized_reduction()` FP64-missing

`codegen/amdgpu/codegen_amdgpu.cpp:160-193` only emits fast reductions for i32/f32. **No FP64 path.** The workload is currently FP32 so this doesn't bite, but will when they go back to FP64. Out of scope for current campaign.

### 7. XCD locality

8 XCDs, kernel launches striped across them. Cross-XCD memory access has higher latency than within-XCD. For persistent kernels or large grids, XCD-aware tiling helps. **Action**: secondary lever; only worth pursuing if the kernel is scaling poorly with grid size.

### 8. `runtime_initialize_rand_states_serial`

Shows up at ≥50% of rocprofv3 profile time. **One-shot init, single call, NOT per-step.** `bench.py` strips it from per-step attribution. Don't waste cycles trying to "fix" it inside a per-step optimization loop — there's a separate ticket to deamortize it (Exp 4 in `prompt_mi300x.md`).

## Profiling tools

### `rocprofv3` (the workhorse)

```bash
# Kernel-stats CSV (what bench.py uses):
rocprofv3 --stats --kernel-trace -d <out_dir> -o <prefix> -- python3 bench_mi300.py ...
# Output: <prefix>_kernel_stats.csv with columns: KernelName, Calls, TotalDurationNs, AverageNs, ...
```

~30% throughput overhead. Always run an untraced bench too for headline numbers.

### `omniperf`

Heavier. Captures occupancy, VGPR/AGPR usage, LDS bank conflicts, L1/L2 hit rates. Use when stuck:

```bash
omniperf profile -n <run_name> -- python3 bench_mi300.py ...
omniperf analyze -p workloads/<run_name>/MI300X --filter <metric>
```

Useful filters: `13.1.6` (LDS bank conflicts), `7.1.0` (VGPR), `7.1.1` (AGPR), `2.1.13` (HBM read BW), `2.1.14` (HBM write BW).

### Dumping the AMDGCN

`quadrants/runtime/amdgpu/jit_amdgpu.cpp` has `print_kernel_llvm_ir` and `print_kernel_amdgcn` config knobs (lines 104, 143). Flip on for one run to dump `quadrants_kernel_amdgpu_llvm_ir_NNNN.ll` and `quadrants_kernel_amdgcn_NNNN.gcn`. Critical for validating: AccVGPR usage, register spill count, `global_load` vs `flat_load`, MFMA emission.

`s_endpgm` ends a kernel. `v_accvgpr_write_b32` / `v_accvgpr_read_b32` indicate AGPR use (you want this for matrix workloads). `scratch_load` / `scratch_store` indicate spill (you don't want this).

## Anti-patterns (do not waste cycles on these)

- **Treating the wave like 32 lanes.** It's 64. CUDA-mask-style `0xFFFFFFFF` is a 32-lane mask; the AMD equivalent is `0xFFFFFFFFFFFFFFFFu` (and even then, only if the AMD shfl backend is wired).
- **Manually unrolling `qd.func` calls hoping the compiler will inline.** Quadrants already handles this; manual unrolls just bloat IR.
- **Tuning `num_stages` / `num_warps` Triton-style on AMD.** Triton-on-ROCm honors them but Quadrants kernels aren't Triton — the equivalent knobs are `block_dim` and `qd.serialize`.
- **Editing the kernel's pytest tolerance.** It's frozen in `bench.py`'s correctness scope; if the agent's edit blew the tolerance, the edit is wrong.
- **Going below `block_dim=64`.** The compiler will silently promote and the result is worse than picking 64 explicitly.
- **Adding `__restrict__`-style hints.** Quadrants doesn't expose them; the compiler-layer is where this lives.
- **Trying to use `cooperative_groups`.** That's CUDA. AMD equivalents go through `__builtin_amdgcn_*` intrinsics — not exposed at the DSL level.

## References

- ROCm CDNA3 ISA reference (gfx942 instruction set): `https://www.amd.com/system/files/TechDocs/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf`
- ROCm performance tuning: `https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x.html`
- Genesis perf tracker (internal): see `project_context.md` for the per-project tracker references.
