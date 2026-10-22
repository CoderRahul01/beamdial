<p align="center">
  <img src="assets/banner.svg" alt="BeamDial: pick Beam's reasoning level per request" width="100%">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-4f8cff" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-3776ab" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/dependencies-zero-3ad0a6" alt="Zero dependencies">
  <a href="https://huggingface.co/reflection/Beam-1.0-501B-A23B"><img src="https://img.shields.io/badge/model-Beam%20501B-a874ff" alt="Model: Beam 501B"></a>
  <a href="https://github.com/CoderRahul01/beamdial/actions/workflows/tests.yml"><img src="https://github.com/CoderRahul01/beamdial/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
  <a href="https://huggingface.co/spaces/PengoSword123/beamdial"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Space-try%20it-ffcc4d" alt="Hugging Face Space"></a>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#results">Results</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="docs/00-start-here.md">Docs</a> ·
  <a href="#reproduce">Reproduce</a>
</p>

---

Beam has five reasoning levels: `low`, `medium`, `high`, `xhigh`, `max`. Most apps pick one and leave it. That either burns thinking tokens on easy prompts or fails the hard ones.

**BeamDial measures what each level actually buys, then picks the level per request.**

- 🎯 **Router.** Predicts P(correct) at every level and picks the cheapest one that clears your target. Pure Python, under 1 MB, runs locally. Your prompt is never sent anywhere to be routed.
- 🪜 **Smart escalation.** Give it a check you already have (tests, a JSON schema, a regex). It climbs one level only when the check fails.
- 🛟 **Production fixes.** A cut-off reply is retried at the same level with more room. Tool calls aren't mistaken for empty answers. Follow-up turns always carry the `reasoning_content` Beam requires.
- 📊 **Open data.** 410 graded tasks × 5 levels = 2,050 runs on Beam, with answers, tokens, latency and correctness.

## Quickstart

```bash
pip install "git+https://github.com/CoderRahul01/beamdial"
```

```python
from openai import OpenAI
from beamdial import BeamDial

beam = BeamDial(OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN))

res = beam.create(messages=[{"role": "user", "content": "How many primes are below 100?"}])
print(res.effort, res.total_tokens, res.content)
```

Escalate only when your own check fails:

```python
res = beam.create(messages=msgs, verify=lambda text: my_tests_pass(text))
```

Keep the conversation going (Beam rejects assistant turns without `reasoning_content`):

```python
msgs.append(res.to_message())
```

**Measure the gap on your own prompts** (the honest answer for your app, not ours):

```bash
# my_prompts.jsonl: {"prompt": "...", "answer": "..."} per line, or pass --check module:function
python -m beamdial calibrate my_prompts.jsonl --levels low,high,max --budget-usd 2
```

It runs each prompt at each level, tests whether the gap is real (exact McNemar test), and prints the cheapest setting that keeps your accuracy, for example `BeamDial(client, policy='cascade', start='low', long_think_tokens=2800, long_think_to='high')`.

See the pick for any prompt, offline:

```bash
python -m beamdial "Find all integers n such that n^2 + 1 divides n^3 + 7"
```

<details>
<summary><b>All options</b></summary>

| Option | Default | What it does |
|---|---|---|
| `policy` | `"router"` | `"router"`, `"cascade"` (start at `start`, climb on failure) or `"fixed"` |
| `target` | `0.85` | Quality target for the router. Higher = safer, more tokens |
| `fallback` | `"medium"` | Level for prompts unlike the training data (not math, not code) |
| `max_effort` | `"max"` | Never go above this level |
| `max_attempts` | `3` | Cap on total calls per request |
| `max_tokens_cap` | `64000` | Ceiling when growing the budget after a cut-off reply |
| `prices` | `None` | `(input, output)` USD per 1M tokens, enables `res.cost_usd` |
| `on_attempt` | `None` | Callback for every attempt, for logging |
| `long_think_tokens` | `None` | If a cheap level thinks longer than this, re-ask once at `long_think_to` (`"high"`). No checker needed |

Every result has `res.attempts`, a list of each call with level, tokens, finish reason and outcome.
</details>

## Results

Every number here comes from `results/summary.json`.

<!-- RESULTS:START -->
**v0.1: 410 tasks × 5 levels = 2,050 graded runs on Beam** (GSM8K, MATH L1 to 5, AMC, AIME 2024, OlympiadBench, HumanEval, MBPP).

<p align="center"><img src="assets/figures/01-hero.png" alt="On standard math and code, Beam's low matches max" width="100%"></p>

| Level | Accuracy | Tokens / task | vs `max` |
|---|---|---|---|
| `low` | 96.3% | 914 | **55% of the tokens** |
| `medium` | 96.1% | 1,236 | 75% |
| `high` | 97.3% | 1,471 | 89% |
| `xhigh` | 97.3% | 1,603 | 97% |
| `max` | 96.6% | 1,657 | 100% |

- **On standard benchmarks `low` is as accurate as `max`**, with all levels within each other's 95% error bars, at about half the tokens.
- The dial does change thinking: on competition math `max` uses 2.2× the tokens of `low`. It just doesn't buy more right answers on these tasks.
- **Free difficulty signal:** when `low` thinks unusually long, re-asking at `high` reached 97.3% at 88% of max's tokens, with no checker needed (`long_think_tokens=3000`).
- 6 of the 8 tasks no level solved are benchmark bugs (listed in `data/flagged_tasks.json`). On the 404 clean tasks, `low` scores 97.8%.

<details>
<summary><b>More charts</b></summary>

<img src="assets/figures/02-dial-works.png" width="100%">
<img src="assets/figures/03-policies.png" width="100%">
<img src="assets/figures/04-long-think-signal.png" width="100%">

Full numbers: [`results/report.md`](results/report.md) · spreadsheet with live formulas: [`results/BeamDial-results-v0.1.xlsx`](results/BeamDial-results-v0.1.xlsx)
</details>

### Where this claim stops

- It's measured on **standard** math and Python benchmarks, where Beam already scores about 97%. On harder problems, more thinking may well pay. That's exactly what v0.2 tests.
- One provider (Baseten via Hugging Face), one sample per cell, temperature 0.7.
- **Don't take our word for your traffic.** Run `python -m beamdial calibrate` on your own prompts (below). It measures the gap for you and prints the setting to use.

**v0.2 (running):** 122 frontier problems (AIME 2026, HMMT 2026, Apex) and 60 long-context state-tracking tasks, where the dial should matter. See [results and roadmap](docs/07-results-and-roadmap.md).
<!-- RESULTS:END -->

## How it works

<p align="center"><img src="assets/how-it-works.svg" alt="How BeamDial routes a request" width="100%"></p>

1. **Measure.** Every task runs at every level on Beam. Math is graded with `math-verify`, code with the benchmark's own tests in a sandbox.
2. **Learn.** One small logistic model per level predicts P(correct | prompt) from hashed word features.
3. **Replay.** Because every task has every level, any strategy (fixed, router, cascade) is scored on real outputs, out-of-fold.
4. **Ship.** The weights are a JSON file inside the library. No runtime dependencies.

Full walkthrough: [docs/02-how-it-works.md](docs/02-how-it-works.md).

## Reproduce

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...                       # your own token, never commit it

python scripts/probe_effort.py               # does reasoning_effort change thinking on each provider?
python scripts/collect.py --pilot            # 14 tasks × 5 levels
python scripts/collect.py --budget-usd 15    # full run, resumable, stops at the budget

# grading runs model-written code: do it in a sandbox (container or VM)
pip install -r requirements-analysis.txt
python scripts/grade.py data/runs.jsonl data/graded.jsonl
python scripts/train_router.py data/graded.jsonl results
python scripts/analyze.py data/graded.jsonl results
```

## Honest limits

- One sample per task per level at temperature 0.7. Runs vary, so small differences mean little.
- Trained on math and Python. Other prompts use the fallback level.
- Token prices differ by provider. Costs use the prices passed to `analyze.py`.
- Some providers may not pass `reasoning_effort` through. The probe checks this.

## Docs

| | |
|---|---|
| [Start here](docs/00-start-here.md) | Map of everything |
| [What it is](docs/01-what-it-is.md) | Problem, users, usage |
| [How it works](docs/02-how-it-works.md) | Every file explained |
| [Edge cases and trade-offs](docs/03-edge-cases-and-tradeoffs.md) | What breaks in production and how we handle it |
| [The probe](docs/04-the-probe.md) | Checking providers before spending |
| [Learning path](docs/05-learning-path.md) | Concepts and papers behind this |
| [Glossary](docs/06-glossary.md) | Every term in plain words |
| [Results and roadmap](docs/07-results-and-roadmap.md) | Findings, targets for v0.2, costs |

## Contributing

PRs welcome, see [CONTRIBUTING.md](CONTRIBUTING.md). Good first issues: async client, streaming, agent and JSON task sets.

## License

Apache 2.0. Built for the [Reflection × Hugging Face "Build with Beam" challenge](https://huggingface.co/reflection/Beam-1.0-501B-A23B/discussions/5).
