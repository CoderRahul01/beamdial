# Edge cases, pain points and trade-offs

This is the part that makes a tool people actually keep using. Anyone can call an API. The value is in the stuff that goes wrong in production.

## What real users run into with Beam

| # | Pain point | What happens | Status in BeamDial |
|---|---|---|---|
| 1 | **Always paying for `max`** | Easy prompts burn thousands of thinking tokens. Slow and expensive. | Handled: router picks per prompt |
| 2 | **Always using `low`** | Hard prompts fail quietly | Handled: router + escalation on failed check |
| 3 | **Reply cut off mid-thought** | `finish_reason="length"`, `content` is `None`. The model spent your budget thinking and never answered. | Handled: retry same level with double budget, up to a cap |
| 4 | **Escalating makes truncation worse** | Naive retry logic climbs to a higher level, which thinks longer and gets cut off again | Handled: truncation never climbs a level |
| 5 | **Tool calls look like empty answers** | Tool-call replies have no text. Naive "empty = failed" logic retries for nothing. | Handled: counted as success |
| 6 | **Multi-turn 400 error** | Beam requires `reasoning_content` on every assistant message you send back. Miss it once and every later request in that session fails. | Handled: `res.to_message()` always includes it, with a placeholder if the provider dropped it |
| 7 | **Runaway retries** | A prompt that always fails the check could climb to `max` again and again | Handled: `max_attempts` (default 3) and `max_effort` caps |
| 8 | **Prompts unlike the training data** | Emails, summaries, chat. The router never saw these and its guess means little. | Handled: detected as "other", uses a fixed `fallback` level (default `medium`) |
| 9 | **Not knowing the real cost** | Failed attempts cost money too. Most wrappers hide them. | Handled: `res.total_tokens` and `res.cost_usd` sum every attempt |
| 10 | **Provider ignores `reasoning_effort`, or your prompts differ from ours** | You think you're saving, nothing changes, or the gap on your traffic is bigger | Handled: `python -m beamdial calibrate` measures it on your own prompts and prints the setting to use |
| 11 | **Privacy of routing** | Hosted routers send your prompt to a third party to decide | Handled: the router runs locally, nothing is sent |
| 12 | **Streaming** | Users want tokens as they arrive | Not yet: a check needs the full answer. Plan: stream passthrough when no `verify` is set |
| 13 | **Async apps** | FastAPI and agent frameworks use async clients | Not yet: `AsyncBeamDial` is on the roadmap |
| 14 | **Wrong `reasoning_effort` values** | `"none"` or `"minimal"` (OpenAI names) return a 400 on Beam | Handled: BeamDial only sends valid Beam levels and rejects a manual `reasoning_effort` kwarg |
| 15 | **Random variation** | Same prompt, same level, different result (temperature 0.7) | Not solvable by a router. We measure it and say so in the report |

## What people dislike about tools like this

- **Magic they can't see.** So every result says which level ran, how many attempts there were, and what each one cost (`res.attempts`).
- **Heavy installs.** So the library has zero required dependencies. The router is plain Python plus a JSON file.
- **Lock-in.** It wraps the standard OpenAI client. Remove BeamDial and the code still works with one line changed.
- **Claims without proof.** So every number comes from graded runs, the data is public, and the pipeline reruns with one command.
- **Surprise bills.** So there are caps on attempts, levels and tokens.
- **Their prompts leaving the building.** So routing is local.

## Decisions we've locked in

| Decision | Why | What we gave up |
|---|---|---|
| Math + Python tasks with known answers | They can be graded automatically and without bias | Coverage of chat, writing and agent tasks |
| One sample per task per level | Fits the $25 budget across 5 levels | Less precise per-task numbers. We report noise honestly |
| Temperature 0.7 | Matches Reflection's published eval settings | Repeat runs vary more than at temperature 0 |
| Logistic regression with hashed features | Tiny, fast, transparent, ships as JSON, no GPU | A fine-tuned small model might route more accurately |
| Pure-Python runtime | Installs anywhere, even in serverless | No numpy speedups (not needed at this size) |
| Monotone smoothing of estimates | Stable, sensible picks | Can't express "higher is worse" even where it really happens |
| Escalation uses only checks a developer can see | Honest simulation | Lower measured gains than using hidden tests would show |
| Grading runs in a sandbox, collection on the Mac | Model-written code never runs on your machine | Extra step to move files across |
| Default target 0.85, fallback `medium`, max 3 attempts | Sensible middle ground | Users with strict needs must tune them |

## The big trade-offs, explained

**Quality vs cost.** The target slider is the whole trade-off. At 0.97 you mostly get `xhigh`/`max` and pay for it. At 0.6 you get cheap levels and accept more misses. There is no free lunch. BeamDial just makes the trade visible and per-prompt.

**One call vs escalation.** Routing alone is one call. Escalation adds calls when a check fails, which costs more on bad prompts but rescues them. If you have a reliable check, escalation from a cheap level is often the best deal. If you don't, rely on the router.

**Accuracy of the router vs simplicity.** A bigger model (say a small transformer) might predict difficulty better. But then the library needs torch, a GPU or a hosted endpoint, and that kills the "drop-in, zero dependency" promise. We pick simple first and only upgrade if the data shows the gap is big.

**General vs specific.** Training on math and code gives clean labels but narrow coverage. The fallback level is the safety valve. Adding agent and JSON tasks is the most valuable next step.

## What the pilot already tells us (and why it matters for the design)

- `low` already solved 13 of 14. Beam is efficient by default.
- Token use barely changed with the level on most tasks. Either the provider ignores the setting, or Beam decides its own thinking budget.
- If the setting doesn't matter, a router has nothing to route. The useful parts become: the finding itself, truncation and multi-turn fixes, and check-based escalation.

That's why the [probe](04-the-probe.md) comes before the full run.
