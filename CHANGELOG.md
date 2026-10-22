# Changelog

## 0.2.0 (in progress)

- Fixed: math grading under-counted correct answers. math-verify can't run in worker threads and the error was counted as wrong. Accuracy 93.1% -> 96.7% on the same runs. Regression tests added.
- New: `long_think_tokens` option. Re-asks once at a higher level when a cheap level thinks unusually long. No checker needed.
- New: majority-vote and length-triggered policies in the analysis; 95% error bars on the main chart.
- New: `data/flagged_tasks.json` lists benchmark tasks with broken reference answers or tests.
- New: `python -m beamdial calibrate prompts.jsonl` measures the reasoning-level gap on your own prompts (exact McNemar test) and prints the cheapest setting that keeps accuracy.
- New: launch figures (3200×1800), results spreadsheet with live formulas, one-command launch script.
- New hard set: frontier math fetcher (AIME 2026, HMMT Feb 2026, Apex) and a synthetic long-context state-tracking generator.

## 0.1.0

- Router: per-level P(correct) with monotone smoothing, cheapest level above a target
- Wrapper: router, cascade and fixed policies; escalation on failed checks; truncation retries at the same level with a bigger budget; tool calls count as success; `to_message()` always carries `reasoning_content`; out-of-domain fallback; attempt, level and token caps; cost estimate
- CLI: `python -m beamdial "prompt"`
- Pipeline: task bank (410 tasks), collector (resumable, budget guard, provider header), graders checked on gold answers, router training with out-of-fold scores, policy replay and charts
- Probe: checks whether `reasoning_effort` changes thinking per provider
