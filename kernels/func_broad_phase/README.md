# Campaign: `func_broad_phase`

Per-campaign directory. The agent reads `target.json` and `playbook.md` at setup, then iterates on Genesis source.

| File | Frozen? | Purpose |
|---|---|---|
| `target.json` | YES | File paths the agent may edit, primary kernel regex, current numbers, correctness scope |
| `playbook.md` | YES | Tier 1-6 ideas seeded from the project tracker. Agent uses as starting point, not a plan |
| `README.md` | YES | This file |

Edits live in `$HOME/work/Genesis/` (worktree per agent), not here.
