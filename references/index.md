# references/

Frozen reference material. **Read once at setup, then consult on demand only**
(e.g. when forming an omniperf-cited hypothesis in B5). Do not re-read every
loop iteration -- that splits your attention budget across files that don't
change between experiments.

| File | Contains | When to consult |
|---|---|---|
| `project_context.md` | Project baselines, top-N kernel gap, layer model, shipped-pattern catalogue. **Per-project** -- replace contents to retarget the harness. | At Phase A setup; on B5 escalation when picking a "shipped pattern from a different campaign" |
| `hardware_notes.md`  | GPU chip details, peak compute, anti-patterns, profiler invocations. **Per-GPU** -- replace contents to retarget. | At Phase A setup; when reasoning about VGPR/AGPR pressure, LDS bank conflicts, occupancy, or profiler output |

The `program.md` operating manual is project-agnostic -- it assumes the agent
will read these two files for project + hardware context. The harness Python
(`bench.py`, `prepare.py`, ...) reads project + platform constants from
[`harness.toml`](../harness.toml) at the repo root, not from these files.

## To retarget at a different project

Both files are templates; the current contents are the Genesis-on-MI300X
instance. To use this harness for a different project:

1. Replace `project_context.md` with your project's baseline numbers,
   baselines, top-N kernel gap, layer model, shipped-pattern catalog.
2. Replace `hardware_notes.md` with your GPU's chip details, peak compute,
   anti-patterns, profiler invocations.
3. Edit `harness.toml` at the repo root for executable values (container
   image, bench script path, profiler command, GPU clock-pin command, etc.).
4. Add per-campaign manifests to `kernels/<your_campaign>/{target.json,
   playbook.md, classes.json}`.

The Python harness and `program.md` should not need any edits.
