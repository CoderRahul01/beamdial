# Checking a provider

Beam is served by several inference providers. Each runs its own copy with its own server settings, and not every one is guaranteed to pass `reasoning_effort` through. Before spending money on a full run, `scripts/probe_effort.py` checks whether the setting actually changes how long Beam thinks on a given provider.

## What it does

It takes 4 hard tasks (easy ones finish in 50 tokens at any level, so they can't show a difference), runs each at `low`, `medium` and `max`, twice, on every provider you list, and prints the median reasoning tokens per provider and level.

```bash
export HF_TOKEN=...
python scripts/probe_effort.py
```

It costs well under $1.

## What we saw (October 22, 2026)

| Provider (via Hugging Face) | low | medium | max |
|---|---|---|---|
| Baseten | 2,974 | 4,026 | 6,994 |
| Together, Fireworks, Nebius | not serving Beam at the time | | |

Median reasoning tokens. On Baseten, `max` thought about 2.35× longer than `low`, so the setting is respected. All runs in this repo are pinned to Baseten (`reflection/Beam-1.0-501B-A23B:baseten`).

An earlier unpinned run, where Hugging Face picked the provider automatically, showed almost no difference between levels on easy tasks. That turned out to be the tasks, not the provider: on easy prompts Beam answers short at every level.

## How to read your own result

| What you see | What it means |
|---|---|
| Tokens rise clearly from low to max (about 2× or more) | The provider respects the setting |
| All levels about the same | Either the provider ignores it, or your tasks are easy. Try harder tasks before concluding |
| ERROR for a provider | It isn't serving Beam right now, or the provider name is different |

One sample per cell is noisy. The same prompt at the same level can use 6k tokens once and 30k the next time, which is why the probe runs each cell twice and reports the median.

## The code, briefly

- `TASK_IDS`: four of the hardest tasks in the set
- `LEVELS = ["low", "medium", "max"]`: the two ends and the middle are enough to see whether the dial moves
- `model:provider` (for example `reflection/Beam-1.0-501B-A23B:together`) pins a provider on the Hugging Face router
- `with_raw_response` reads the `x-inference-provider` header, proof of which provider actually answered
- a provider that returns 400/401/403/404/422 is skipped for the remaining calls
- every raw result is saved to `data/probe_runs.jsonl`
