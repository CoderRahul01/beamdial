# Results so far and the road to v0.2

## v0.1 in one paragraph

We ran 410 tasks at all 5 reasoning levels on Beam (Baseten via Hugging Face), 2,050 graded runs, for about $3.50. **On standard benchmarks, `low` is as accurate as `max` (96.3% vs 96.6%, no significant difference) for 55% of the tokens.** The dial works: on competition math, `max` uses 2.2× the tokens of `low`. But on these tasks, the extra thinking doesn't buy extra right answers. Most of the remaining "failures" are bugs in the benchmarks themselves, not Beam.

## The numbers

| Level | Accuracy (410 tasks) | Accuracy (404 clean tasks) | Tokens per task | vs max |
|---|---|---|---|---|
| low | 96.3% | 97.8% | 914 | 55% of the tokens |
| medium | 96.1% | 97.5% | 1,236 | 75% |
| high | 97.3% | 98.8% | 1,471 | 89% |
| xhigh | 97.3% | 98.8% | 1,603 | 97% |
| max | 96.6% | 98.0% | 1,657 | 100% |

- The 95% error bar on each accuracy is about ±2 points, so all five levels are statistically tied.
- Head to head, low vs max: low wrong but max right on 4 tasks, low right but max wrong on 3. Basically a coin flip (p = 1.0).
- 384 of 410 tasks were solved at **every** level. Only 18 tasks flip between levels.
- Thinking does scale with the level where it can: on AMC/AIME/Olympiad, mean tokens go from 1,756 at `low` to 3,873 at `max`. On easy tasks, Beam ignores the dial and answers short.

## Three things we found that nobody else has published

1. **`low` is enough for standard math and code on Beam.** Using `max` by default costs about 1.8× the tokens for the same accuracy.
2. **Beam's own thinking length is a free difficulty signal.** On the 23 tasks where `low` spent more than 3,000 reasoning tokens, `low` was right 83% of the time, `high` 100% and `max` 87%. A check-free policy, "use `low`, re-ask at `high` if low thought long", hit 97.3% at 88% of max's tokens, tied with the best fixed level. It's small data (23 such tasks), so v0.2 must confirm it.
3. **6 of the 8 tasks no level could solve are benchmark bugs.** One GSM8K reference answer is wrong. Five MBPP hidden tests demand things the prompt never states (return `None` not `False`, a specific permutation order). All listed in `data/flagged_tasks.json`.

## Why this holds up (and where it stops)

People will push back with "your tasks are too easy, on frontier problems `max` wins". That may be true, and we say so. The claim is built to survive that:

1. **Scoped claim.** Every headline says "on standard math and code". We never say "you never need max".
2. **Uncertainty shown.** Every accuracy has a 95% interval, and low vs max has an exact significance test (p = 1.0).
3. **Adaptive by design.** BeamDial doesn't lock you to `low`. The long-think signal and check-based escalation move up when a prompt is hard. If frontier problems need `max`, the policy reaches for it.
4. **Everything public.** All 2,050 raw outputs, grades and code, so anyone can find a mistake. We already found and fixed one of our own.
5. **Your prompts, your answer.** `python -m beamdial calibrate` measures the gap on the user's own traffic in minutes and prints the setting to use. Our benchmark is a starting point, not the final word.
6. **v0.2 attacks our own claim.** Frontier math (AIME 2026, HMMT 2026, Apex) and fresh long-context tasks are chosen where the gap is most likely to appear.

## A bug we caught in our own grader

The first grading said 93.1%. The real number is 96.7%. The math checker (`math-verify`) refuses to run inside worker threads, and our grader quietly counted that as a wrong answer. So `\frac14` vs `\frac{1}{4}` was marked wrong. Fixed: math is graded in the main thread, grader errors are labelled instead of hidden, and 11 regression tests guard it. Every number above uses the fixed grader. Re-grading independently gave identical results.

This is worth saying publicly. "Here's a bug we found in our own pipeline and how" builds more trust than "everything worked".

---

## What v0.2 has to prove (the bar to cross)

v0.1's tasks are too easy for Beam, so the dial can't show its value. v0.2 adds tasks where Beam isn't at 97%.

| # | Target | Pass if | Why it matters |
|---|---|---|---|
| 1 | **Find where the dial matters** | On the hard set, the best level beats `low` by ≥ 8 points, with p < 0.05 | Proves when `max` is worth paying for |
| 2 | **BeamDial beats fixed levels** | A BeamDial policy gets within 1 point of the best fixed level at ≤ 70% of its tokens (out-of-fold) | Proves the tool saves money without losing quality |
| 3 | **Confirm the free signal** | "Low, re-ask at high if it thinks long" stays on or above the fixed-level line on the hard set | The headline feature works with no checker |
| 4 | **Honest null result, if it comes** | If even frontier problems show no gap, publish that as the finding | "You never need max on Beam" is a big claim and still a win |

## The v0.2 hard set (built, ready to run)

| Set | Tasks | What it tests | Contamination risk |
|---|---|---|---|
| AIME 2026 | 30 | Recent competition math | Before Beam's cutoff, may be seen |
| HMMT Feb 2026 | 33 | Harder competition math | Before cutoff, may be seen |
| Apex 2025 + Apex Shortlist | 59 | Problems picked to be hard for frontier models | Before cutoff, may be seen |
| **Long-context state tracking** | 60 | Follow variable updates through 8k to 32k token logs, latest value wins | **None**: generated fresh, can't be memorized |

The long-context set is our own invention. It tests exactly what breaks production agents: tracking state that changes over time across a long context (the stale-memory problem). It's also where Reflection's own blog showed `low` failing. Difficulty is dialed by log length (8k / 32k tokens) and number of updates (25 / 60 / 120). Every gold answer is computed by executing the log, then re-checked by an independent replay.

## How we push accuracy up (the levers)

| Lever | Cost | Expected effect | Status |
|---|---|---|---|
| Fix grader false negatives | Free | +3.6 points of real accuracy | Done |
| Flag broken benchmark tasks | Free | Clean ceiling is about 98 to 99% | Done |
| Length-triggered re-ask (low → high) | Free to simulate; in use about 40 to 60% more tokens than plain `low`, still under `max` | +1 point on v0.1, likely more on hard tasks | Built into the library (`long_think_tokens=3000`) |
| Majority vote of 3 at `low` | 3× low's tokens | Often beats one `max` run on hard math | Simulation ready, needs `--samples 3` at low |
| Escalate on a failed check | Small | +0.5 points on v0.1 | Done |
| Router trained on hard + easy data | Free after collection | Real routing instead of "always low" | Retrain after v0.2 |

## What it will cost

Measured on your run: Baseten via Hugging Face, AIME 2024 averaged about 4,950 output tokens per call across levels. The frontier sets are harder, so the range assumes 1× to 2.5× that. Long-context costs are mostly input tokens (about 1.4M per level pass). Prices are $0.40 per 1M input and $1.20 per 1M output (the script's defaults). **Check your real charges on the Hugging Face billing page**, since the provider's actual price may differ.

| Plan | What runs | Calls | Estimated cost |
|---|---|---|---|
| Pilot | 2 tasks per source × 5 levels | 60 | $0.50 to $0.90 |
| A. Lean | Hard set × 5 levels, 1 sample | 910 | $7 to $14 |
| **B. Solid (recommended)** | A + 2 extra `low` samples on frontier math (for majority vote) | 1,154 | **$9 to $17** |
| C. Strong | 2 samples everywhere + vote samples | 2,064 | $16 to $31 |

Spent so far (estimated): pilot $0.22 + probe $0.15 + full run $3.51 ≈ **$3.90**. That leaves roughly **$21** of the $25. The live version of this math is the `v0.2 cost plan` sheet in `results/BeamDial-results-v0.1.xlsx`; edit its yellow cells.

**Recommendation: Pilot, then Plan B, with `--budget-usd` set as a hard stop.** The collector halts at the budget and resumes later, so overspending isn't possible beyond the calls already in flight.

## Run order for v0.2

```bash
cd "/Volumes/Powerhouse/Agentic Systems/beamdial"
export HF_TOKEN=...   # never paste this line anywhere
M=reflection/Beam-1.0-501B-A23B:baseten

python3 scripts/fetch_frontier.py                     # downloads AIME 2026, HMMT 2026, Apex -> data/tasks_frontier.jsonl
python3 scripts/build_longctx.py                      # already built, re-creates the same 60 tasks

# pilot (about $0.50 to $0.90): paste the summary table
python3 scripts/collect.py --pilot --tasks data/tasks_frontier.jsonl --out data/pilot_frontier.jsonl --model $M --max-tokens 64000
python3 scripts/collect.py --pilot --tasks data/tasks_longctx.jsonl  --out data/pilot_longctx.jsonl  --model $M

# Plan B (only after the pilot looks sane)
python3 scripts/collect.py --tasks data/tasks_frontier.jsonl --out data/runs_frontier.jsonl --model $M --max-tokens 64000 --budget-usd 11
python3 scripts/collect.py --tasks data/tasks_frontier.jsonl --out data/runs_frontier.jsonl --model $M --max-tokens 64000 --levels low --samples 3 --budget-usd 13
python3 scripts/collect.py --tasks data/tasks_longctx.jsonl  --out data/runs_longctx.jsonl  --model $M --budget-usd 5
```

`--budget-usd` counts everything already in that output file, so the second frontier command stops at $13 total for that file, not $13 more. Grading, training and charts happen in my sandbox: send the run files over and I'll do the rest.

## What's still missing (honest list)

- **Agent and tool-calling tasks.** That's where Beam is mostly used, and where the level may matter most. Next after v0.2.
- **Code at frontier difficulty** (LiveCodeBench hard). Its test harness is heavy, so it's planned for after the hackathon.
- **One provider only.** Only Baseten serves Beam on HF right now. Other providers (or Reflection's own API) could behave differently.
- **Async client and streaming** in the library.
