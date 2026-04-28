# Compiler-output patterns

Calibration examples for the agent. When `bench.py --dump-amdgcn` produces an `.ll` file (LLVM IR targeted at `amdgcn-amd-amdhsa`), use these examples to pattern-match what's interesting in YOUR kernel's dump.

**This doc is a calibration aid, not a checklist.** The whole point of the dump is that Claude can spot inefficiencies the harness doesn't pre-detect. Use these examples to learn what AMDGCN-flavored LLVM IR looks like, then look for these AND anything else that looks suboptimal in the kernel you're optimizing.

## How to read a dump

After `bench.py --dump-amdgcn`, the output prints lines like:

```
amdgcn dump: /work/runs/<campaign>-<id>/amdgcn/T0bfec00c4da1.ll
amdgcn dump: /work/runs/<campaign>-<id>/amdgcn/T1957519deed51.ll
...
```

Each `.ll` is the LLVM IR for one compiled kernel. The filenames are content-hashed, so you can't tell which is which by name. Find the one for YOUR kernel by greping for the kernel name:

```bash
docker exec gbench bash -c "grep -l 'func_broad_phase_c' /work/runs/<campaign>-<id>/amdgcn/*.ll"
```

Then read just that file. A typical kernel IR is 200-2000 lines.

The IR is compiler output, not source. It has been:
- Inlined past `qd.func` boundaries (you see all the work in one function body)
- Optimized by Quadrants' middle-end (loop transforms, scalar promotions)
- Annotated with `amdgpu_kernel` calling convention and `addrspace` qualifiers

This is exactly the level where Move-37–style insights live: you can see what the compiler chose to emit, including the things the source-level intent didn't anticipate.

## What to look for (calibration examples)

### 1. Register pressure / spill markers

In LLVM IR, register pressure shows up as **lots of `alloca` in `addrspace(5)` (scratch)** for non-trivial-sized objects, and stores/loads through those `addrspacecast` chains.

```llvm
%14 = alloca i64, align 8, addrspace(5)
%15 = addrspacecast ptr addrspace(5) %14 to ptr
...
store i64 %someval, ptr %15, align 8
%16 = load i64, ptr %15, align 8
```

A function with **dozens of `alloca`s in addrspace(5)** is using a lot of scratch — the AMDGCN backend will likely emit `scratch_load_*` / `scratch_store_*` instructions, which spill from registers to local memory. Your hypothesis: reduce the live set by reordering computation, narrowing types, or hoisting common subexpressions out of the high-pressure region.

### 2. Address-space mismatches (slow flat loads)

LLVM IR distinguishes:
- `addrspace(0)` — generic / flat (slowest; uses `flat_load` instructions)
- `addrspace(1)` — global memory (fast `global_load_*`)
- `addrspace(3)` — LDS / shared memory
- `addrspace(4)` — constant
- `addrspace(5)` — scratch / private

Look for **loads/stores on `ptr` (no addrspace, i.e. addrspace(0))** that you'd expect to be on `ptr addrspace(1)` (global). Each one is a missed `addrspacecast` upstream:

```llvm
; SLOW (flat):
%val = load float, ptr %p, align 4

; FAST (global):
%pg = addrspacecast ptr %p to ptr addrspace(1)
%val = load float, ptr addrspace(1) %pg, align 4
```

Hypothesis: find where the `ptr` originated (often `getelementptr` from a struct member that lost its addrspace through a cast or function boundary), restore the addrspace cast in the source.

### 3. Missed vectorization

A sequence like:

```llvm
%a0 = getelementptr float, ptr addrspace(1) %base, i64 0
%v0 = load float, ptr addrspace(1) %a0, align 4
%a1 = getelementptr float, ptr addrspace(1) %base, i64 1
%v1 = load float, ptr addrspace(1) %a1, align 4
%a2 = getelementptr float, ptr addrspace(1) %base, i64 2
%v2 = load float, ptr addrspace(1) %a2, align 4
%a3 = getelementptr float, ptr addrspace(1) %base, i64 3
%v3 = load float, ptr addrspace(1) %a3, align 4
```

is four scalar loads at consecutive offsets. The backend MIGHT coalesce into one `global_load_dwordx4`, but often doesn't if the loads are interleaved with arithmetic or come through different SSA values. Hypothesis: at the source, declare the read as `vec4` (or whatever the DSL equivalent is for your project) so the IR emits a single wide load directly.

### 4. AMD-specific intrinsics worth knowing

These are the GPU primitives Quadrants uses. Their presence/absence tells you what the agent has access to:

| Intrinsic | Meaning |
|---|---|
| `@llvm.amdgcn.workitem.id.x()` | thread index within wave |
| `@llvm.amdgcn.workgroup.id.x()` | block index within grid |
| `@llvm.amdgcn.dispatch.ptr()` | kernel argument pointer |
| `@llvm.amdgcn.implicitarg.ptr()` | implicit kernel args (workgroup size etc.) |
| `@llvm.amdgcn.s_barrier()` | wave-level barrier |
| `@llvm.amdgcn.ds.bpermute()` | data sharing across lanes via LDS |
| `@llvm.amdgcn.update.dpp()` | data-parallel primitive (cross-lane shuffle) |
| `@llvm.amdgcn.mfma.*` | matrix-fused-multiply-add (MFMA) — peak compute |
| `@llvm.amdgcn.global.load.lds.*` | async global → LDS copy (overlap with compute) |

Things to flag:
- A kernel that's MFMA-eligible (matrix arithmetic) but has no `@llvm.amdgcn.mfma.*` calls — the compiler missed it; restructure to expose matmul.
- Manual reduction loops with no `@llvm.amdgcn.ds.bpermute` or DPP — could replace O(N) serial reduction with O(log N) lane-shuffle.
- Sequential global loads followed by computation, with no `@llvm.amdgcn.global.load.lds` — could overlap memory with compute via async LDS staging.

### 5. Kernel boundary / size

```llvm
define amdgpu_kernel void @func_broad_phase_c400_0_kernel_2_range_for(...) #0 {
```

The function attributes (`#0`) at the bottom of the file tell you what register / occupancy hints LLVM saw:

```llvm
attributes #0 = { "amdgpu-flat-work-group-size"="64,64" "amdgpu-waves-per-eu"="2,2" ... }
```

`amdgpu-waves-per-eu="2,2"` means 2 waves per execution unit max — that's ~20% peak occupancy on CDNA3. If the kernel is memory-bound, this is the single biggest knob. (Note: per `references/hardware_notes.md` this is currently a uniform compiler-layer setting, not per-kernel — but it's worth flagging.)

Total instruction count (proxy: `wc -l` of the function body) tells you how big the kernel got. A 2000-line LLVM IR function is huge — probably aggressive inlining; consider whether some `qd.func` calls should NOT have been inlined.

### 6. Inlined function fragments — look for surprise

Quadrants inlines `qd.func` calls aggressively. The IR shows the result. Sometimes inlining produces emergent inefficiencies the source-level review couldn't catch:

- Two inlined helpers each load the same `getelementptr` independently (CSE missed)
- An inlined branch with a loop-invariant condition that didn't get hoisted
- Two-step `addrspacecast` (`ptr → ptr addrspace(5) → ptr`) that round-trips for no reason
- Silent type promotions (`i32 → i64 → i32`) that produce extra instructions

Read 200-500 lines of the kernel body and just look for things that "feel wrong." This is where the LLM's pattern-matching, applied to off-distribution input (your specific kernel's IR), has the best shot at Move 37.

### What you should NOT do

- **Don't try to recompile the IR.** It's extracted from a Quadrants cache file (`.tic`) via `strings`; some lines may be missing. It's for reading, not feeding back to clang.
- **Don't blindly enumerate every pattern in this doc.** That degenerates back into checklist mode. Pick the most striking thing in your specific dump and form a focused hypothesis.
- **Don't read all 78 dump files.** Grep for your kernel name, read only that file.

## When to consult this

- During Phase B4 (when stuck) — the AMDGCN dump is the FIRST escalation path.
- During exploration iterations (B6 generator 1) — a periodic excuse to look at the IR even when not stuck.
- After any kept experiment whose mechanism you don't fully understand — the IR will tell you what actually changed.
