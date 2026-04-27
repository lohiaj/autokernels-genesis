# Campaign: `kernel_step_1_2`

Per-campaign directory. The agent reads `target.json` and `playbook.md` at setup, then iterates on Genesis source.

| File | Frozen? | Purpose |
|---|---|---|
| `target.json` | YES | File paths the agent may edit, primary kernel regex (matches both kernel_step_1 and kernel_step_2 sub-kernels), correctness scope |
| `playbook.md` | YES | Tier 1-6 ideas, including the proven `factor_mass` reference pattern |
| `README.md` | YES | This file |

Edits live in `~/work/Genesis/` (worktree per agent), not here.
