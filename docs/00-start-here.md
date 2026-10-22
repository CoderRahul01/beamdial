# Start here

BeamDial picks how hard Beam should think for each request. Easy prompts get a cheap setting, hard prompts get more thinking, and you only pay for `max` when it actually helps.

Read these in order. Each one is short.

| # | Doc | What you get from it |
|---|---|---|
| 1 | [What it is](01-what-it-is.md) | The problem, who it's for, what users get, how they use it |
| 2 | [How it works](02-how-it-works.md) | The full pipeline and a walkthrough of every file and its key lines |
| 3 | [Edge cases and trade-offs](03-edge-cases-and-tradeoffs.md) | What breaks for real users, what we handle, what we decided and why |
| 4 | [The probe](04-the-probe.md) | What the probe test is, line by line, and how to read its result |
| 5 | [Learning path](05-learning-path.md) | The concepts behind this project and what to study next |
| 6 | [Glossary](06-glossary.md) | Every term used here, in plain words |
| 7 | [Results and roadmap](07-results-and-roadmap.md) | What v0.1 found, the bar for v0.2, how to raise it, what it costs |

Your launch plan and post drafts are in `private/launch-and-marketing.md`. That folder is in `.gitignore`, so it never goes to GitHub.

## The one-minute version

- Beam (Reflection's 501B open model) has a setting called `reasoning_effort` with 5 values: `low`, `medium`, `high`, `xhigh`, `max`.
- Higher means more hidden "thinking" tokens before the answer. Thinking tokens are billed like output tokens and add wait time.
- Nobody has published which level you need for which kind of task. So people either overpay (always `max`) or fail hard tasks (always `low`).
- We run 410 tasks with known answers at all 5 levels, measure right or wrong, tokens and time, and train a tiny router on that data.
- Developers wrap their existing OpenAI client with BeamDial. It picks the level per request, retries smartly when a check fails, and handles the API quirks that break Beam apps.

## Where things are

```
beamdial/            the library people install (router + wrapper + CLI)
scripts/             the research pipeline (build tasks, collect, grade, train, analyze, probe)
data/tasks.jsonl     the 410 tasks
space/               the Hugging Face Space (demo app)
tests/               unit tests and a fake Beam server for offline testing
docs/                you are here
```

## Status (Oct 22)

- v0.1 done: 410 tasks x 5 levels on Beam (Baseten), 2,050 graded runs, about $3.50.
- Finding: `low` matches `max` within noise (96.3% vs 96.6%) at 55% of the tokens. See [results and roadmap](07-results-and-roadmap.md).
- v0.2 built and ready to run: 122 frontier math problems + 60 long-context state-tracking tasks.
