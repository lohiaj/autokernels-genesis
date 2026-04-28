# references/

Frozen reference material. **Read once at setup, then consult on demand only**
(e.g. when forming an omniperf-cited hypothesis in B5). Do not re-read every
loop iteration -- that splits your attention budget across files that don't
change between experiments.

| File | Contains | When to consult |
|---|---|---|
| `project_context.md` | Project baselines, top-8 kernel gap, layer model, shipped-pattern catalogue | At Phase A setup; on B5 escalation when picking a "shipped pattern from a different campaign" |
| `mi300x_notes.md` | CDNA3/gfx942 hardware cheatsheet, anti-patterns, profiler invocations | At Phase A setup; when reasoning about LDS bank conflicts, register pressure, occupancy, or profiler output |

The `program.md` operating manual is the only doc you need open during the
loop. The orchestration tools (`orchestrate.py`, `summarize.py`, `global_log.py`,
`watchdog.py`) and the per-campaign `kernels/$CAMPAIGN/{target.json,playbook.md}`
are also load-bearing -- everything else is reference.
