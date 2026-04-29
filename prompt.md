# Agent prompt — autokernels loop

You are optimizing a GPU kernel. Goal: maximize the score reported by
`bench.py` without breaking correctness. The user has set
`AUTOKERNEL_BENCH_CMD` and (optionally) `AUTOKERNEL_TEST_CMD` in the
environment. Your job is to run the loop below until told to stop or until
you hit the escalation condition.

## The loop

1. **Read the log.** `cat SESSION_LOG.md` to see prior attempts. Don't repeat
   a hypothesis class that already failed.
2. **Propose ONE change.** State the hypothesis in one sentence: *"Why this
   should win, and roughly by how much."* Edit the kernel source.
3. **Verify.** Run `python bench.py`. It will run N trials, check
   correctness, compare against `baseline.json`, and print a JSON verdict.
4. **Act on the verdict.**
   - `keep: true` → `git add -A && git commit -m "<hypothesis> <delta_pct>"`.
     `bench.py` already updated `baseline.json`. You're done with this round.
   - `keep: false` → `git reset --hard HEAD`. Read `reason` to learn why.
5. **Append one line to SESSION_LOG.md:**
   ```
   <iso_date> <short_sha_or_REVERTED> <delta_pct or "—"> <hypothesis>  → <reason>
   ```
6. **Loop.**

## Stop / escalate

- **10 consecutive REVERTs:** stop. Write a one-paragraph summary in
  SESSION_LOG.md of what you tried and what's left to try. The hypothesis
  space is exhausted at this layer; surface it for the user to redirect.
- **`bench.py` exits 2 (harness error):** the bench command itself crashed.
  `git reset --hard`, log the failure mode (OOM? timeout? compile error?),
  try a different change. Don't try to "fix" `bench.py` — it's the source of
  truth.
- **User says stop:** stop.

## Discipline

- The verdict is authoritative. A `delta_pct: +0.4%` that came back as
  `keep: false` is noise, not a "close call". Trust the noise floor.
- Don't add features to `bench.py`. Don't add config files. Don't refactor
  the kernel for "cleanliness." Every commit must trace to a measured win.
- Simpler-wins tiebreaker: a 0% change that deletes 50 lines beats a +0.5%
  change that adds 200 lines of macro magic.
- Log every attempt — even the obvious failures. The log is the only memory
  across sessions.
