# Changelog

## 0.1.0 (2026-10-22)

First release.

- Study: 410 tasks at all five reasoning levels on Beam, 2,050 graded runs, with policies replayed on the real outputs and 95% intervals on every accuracy.
- Library: router, cascade and fixed policies; escalation on a failed check; same-level retry with a bigger budget for cut-off replies; tool calls count as success; `to_message()` always carries `reasoning_content`; out-of-domain fallback; attempt, level and token caps; cost estimate.
- `beamdial` command: local pick for a prompt, and `beamdial calibrate` to measure the gap on your own prompts.
- `long_think_tokens` option, off by default. It ties "always high" on v0.1 data and is kept for testing on harder sets.
- Demo as a static page with the router ported to JavaScript and checked against the Python version on every task.
- Grader fix: math-verify can't run in worker threads, and an early version counted that as a wrong answer. Math is now graded in the main thread, with regression tests.
- `data/flagged_tasks.json` lists 6 benchmark items with broken reference answers or tests.
- Hard set ready to run: frontier math fetcher (AIME 2026, HMMT Feb 2026, Apex) and a synthetic long-context state-tracking generator.
