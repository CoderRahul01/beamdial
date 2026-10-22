---
pretty_name: BeamDial v0.1 - Beam reasoning effort benchmark
license: other
license_name: mixed-source
license_link: LICENSE
language:
- en
tags:
- beam
- reflection-ai
- reasoning
- reasoning-effort
- test-time-compute
- llm-routing
- benchmark
task_categories:
- text-generation
size_categories:
- 1K<n<10K
configs:
- config_name: runs
  data_files: data/runs_graded.jsonl
  default: true
- config_name: tasks
  data_files: data/tasks.jsonl
---

# BeamDial v0.1: what each reasoning level buys on Beam

![BeamDial results](figures/01-hero.png)

**2,050 graded runs**: 410 tasks with known answers, each run at all five `reasoning_effort` levels (`low`, `medium`, `high`, `xhigh`, `max`) on Reflection's [Beam 1.0 501B A23B](https://huggingface.co/reflection/Beam-1.0-501B-A23B).

| Level | Accuracy | 95% interval | Mean completion tokens | vs max |
|---|---|---|---|---|
| low | 96.3% | 94.1 to 97.8% | 914 | 55% |
| medium | 96.1% | 93.8 to 97.6% | 1,236 | 75% |
| high | 97.3% | 95.3 to 98.5% | 1,471 | 89% |
| xhigh | 97.3% | 95.3 to 98.5% | 1,603 | 97% |
| max | 96.6% | 94.4 to 98.0% | 1,657 | 100% |

**Finding:** on standard math and code, all five levels are a statistical tie on accuracy. `low` uses 55% of `max`'s tokens. The dial does change thinking (on competition math, `max` reasons 2.2× longer than `low`); it just doesn't buy more right answers on these tasks.

## What's inside

| Config / file | Rows | What it is |
|---|---|---|
| `runs` (`data/runs_graded.jsonl`) | 2,050 | One row per (task, level): Beam's answer, its reasoning, token usage, latency, finish reason, and the grade |
| `tasks` (`data/tasks.jsonl`) | 410 | The prompts and gold answers (or test code) |
| `data/flagged_tasks.json` | 6 | Tasks with broken reference answers or tests, with the reason for each |
| `results/summary.json` | | Every number in the report, including replayed routing policies |
| `results/BeamDial-results-v0.1.xlsx` | | Spreadsheet with live formulas |
| `figures/` | | Launch figures (PNG, 3200×1800) |

Key fields in `runs`: `task_id`, `source`, `family`, `effort`, `content`, `reasoning`, `finish_reason`, `latency_s`, `usage.prompt_tokens`, `usage.completion_tokens`, `usage.reasoning_tokens`, `correct`, `grade_reason`, `pred`, `public_pass`, `served_by`.

## How it was made

- **Tasks:** GSM8K (40), MATH levels 1 to 5 (160), AMC 2023 (40), AIME 2024 (30), OlympiadBench numeric (40), HumanEval (50), MBPP test split (50). Sampled with a fixed seed.
- **Model and serving:** `reflection/Beam-1.0-501B-A23B` through Hugging Face Inference Providers, pinned to Baseten. Temperature 0.7, `max_completion_tokens` 32,000, one sample per task per level, October 22, 2026.
- **Grading:** math via [math-verify](https://github.com/huggingface/Math-Verify) on the last `\boxed{}` answer; code by running the benchmark's own tests in an isolated process with a timeout. `public_pass` records only the test a developer could see (the MBPP example or the HumanEval docstring examples).
- **Code:** everything to reproduce, re-grade and re-chart is at https://github.com/OWNER/beamdial

## Known limits

- One sample per cell at temperature 0.7, so per-task results are noisy. Compare levels using the intervals above.
- Math and Python only. Agent, tool-calling and long-context tasks are coming in v0.2.
- One provider (Baseten). Other servers may treat `reasoning_effort` differently.
- 6 tasks are flagged as broken benchmark items (one wrong GSM8K reference, five underspecified MBPP tests). Excluding them, `low` scores 97.8%.
- These benchmarks are old and public, so Beam has likely seen many of them in training. That can make `low` look better than it would on new problems.
- Median latency was about 1.1 s at every level. Means differ only because of a long tail of hard prompts.
- The "re-ask at high when low thinks long" policy in `results/summary.json` uses a threshold picked on this data. With the threshold picked on held-out folds it scores 97.2%, a tie with always using `high`.
- An early version of our grader under-counted math answers (math-verify cannot run in worker threads). This release uses the fixed grader. The fix and regression tests are in the repo.

## License

The prompts and reference answers come from the public benchmarks listed above and keep their original licenses and terms. Beam's outputs, our grades, annotations and code are released under Apache 2.0. Reflection's model card permits use of Beam outputs, including for training other models.

## Citation

```bibtex
@misc{beamdial2026,
  title  = {BeamDial: what each reasoning level buys on Beam},
  author = {Rahul Pandey},
  year   = {2026},
  url    = {https://huggingface.co/datasets/HFUSER/beamdial-v0.1}
}
```
