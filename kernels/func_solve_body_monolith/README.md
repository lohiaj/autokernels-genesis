# Campaign: `func_solve_body_monolith`

The dominant 54% GPU-time kernel. The single biggest e2e leverage point.

| File | Frozen? | Purpose |
|---|---|---|
| `target.json` | YES | Regex `^func_solve_body_monolith_c\d+_\d+_kernel_\d+`, edit scope (solver.py + abd/forward_dynamics.py), correctness tests |
| `playbook.md` | YES | Tier 1-5 ideas including team-shipped patterns, prior-session wins (exp 21, 22, 28, 32), and proven anti-patterns |
| `README.md` | YES | This file |

The agent edits `~/work/Genesis/genesis/engine/solvers/rigid/constraint/solver.py` and `abd/forward_dynamics.py`. Quadrants compiles `@qd.kernel func_solve_body_monolith` and inlines all `@qd.func` callees into one monolith kernel — so edits to `func_solve_iter`, `func_linesearch_batch`, `func_update_constraint_batch`, `func_update_gradient_batch`, `func_terminate_or_update_descent_batch`, and `func_solve_mass_entity` all directly affect this kernel's compiled output.
