# The probe

## What it is, in one line

A small, cheap experiment (under $1) that checks one thing before we spend real money: **does `reasoning_effort` actually change how much Beam thinks, and does that depend on which company is serving Beam?**

## Why we need it

Beam isn't served by one company. Through Hugging Face, your request goes to one of several inference providers: Together, Fireworks, Baseten, Nebius and others. Each runs its own copy of Beam with its own server settings. Some might pass `reasoning_effort` through correctly. Some might ignore it.

In the pilot, HF picked a provider for us automatically and the token counts barely changed from `low` to `max`. Two explanations fit:

1. **That provider ignores the setting.** Then the full run would measure the same thing 5 times and waste credits.
2. **Beam picks its own thinking budget**, and the setting only nudges it. Then that is the finding.

The probe tells these apart by **pinning** each provider and comparing them side by side.

## How to read the result

The script prints a table like this (made-up numbers):

```
target             served_by          low    medium       max   (median reasoning tokens)
hf:together        together         2,400     2,500     2,450
hf:fireworks-ai    fireworks-ai       900     2,800    11,000
hf:baseten         -                    ...
```

| What you see | What it means | What we do |
|---|---|---|
| One provider's numbers rise clearly from low to max | That provider respects the setting | Do the full run pinned to that provider |
| All rows flat | Beam mostly ignores the setting on HF | Lead with the finding, drop the router, ship the reliability layer |
| A row shows ERROR | That provider doesn't serve Beam right now, or the name is different | Ignore it, look at the others |
| Numbers rise only on the hardest task | The setting matters only for very hard problems | Full run still worth it, the router learns "hard = more" |

"Clearly" means max uses at least about 2x the tokens of low. One sample per cell bounces around, so small differences mean nothing.

## The code, line by line

```python
TASK_IDS = ["aime24-73", "math-precalculus-1081", "olympiad-2759", "olympiad-2745"]
```
Four of the hardest pilot tasks. Easy tasks finish in 50 tokens at any level, so they can't show a difference. Hard ones can.

```python
LEVELS = ["low", "medium", "max"]
```
Three levels, not five. The two ends plus the middle is enough to see whether the dial moves, at 60% of the cost.

```python
SAMPLES = 2
```
Each cell runs twice. We already saw one AIME task use 6k tokens once and 30k another time at nearly the same setting. Two samples and a median smooth that a bit.

```python
HF_PROVIDERS = ["together", "fireworks-ai", "baseten", "nebius"]
```
Providers to pin. On the HF router, `model:provider` forces a specific provider, for example `reflection/Beam-1.0-501B-A23B:together`.

```python
def targets():
    if os.environ.get("HF_TOKEN"): ... one target per HF provider
    if os.environ.get("BEAM_API_KEY"): ... plus Reflection's own API
```
Builds the list of places to test. Reflection's API is the "reference" version, served by the people who made the model. Its key is optional.

```python
jobs = [(name, c, m, tid, lvl, s) for name, c, m in targets() for tid in TASK_IDS for lvl in LEVELS for s in range(SAMPLES)]
```
Every combination: 4 providers x 4 tasks x 3 levels x 2 samples = 96 calls (120 with Reflection's API).

```python
raw = c.chat.completions.with_raw_response.create(model=model, messages=[...], reasoning_effort=lvl,
                                                  max_completion_tokens=32000, temperature=0.7)
```
The call. `max_completion_tokens=32000` leaves room for long thinking, so `max` isn't cut off early and made to look cheap. `temperature=0.7` matches the main study.

```python
served = raw.headers.get("x-inference-provider") or raw.headers.get("x-provider")
```
Reads which provider actually answered, from the response headers. Proof the pin worked.

```python
if status in (400, 401, 403, 404, 422): dead.add(name)
```
If a provider rejects us (wrong name, not serving Beam), stop sending it the remaining calls. That saves time and money.

```python
with ThreadPoolExecutor(6) as ex:
```
Six requests at a time. Fast enough, and gentle enough not to hit rate limits.

```python
f.write(json.dumps(r) + "\n")
```
Every raw result is saved to `data/probe_runs.jsonl`, so we can dig deeper later (accuracy per provider, latency) without paying again.

```python
meds = [statistics.median([(r["usage"]["reasoning_tokens"] or r["usage"]["completion_tokens"]) for r in ok if r["effort"] == l]) for l in LEVELS]
```
The summary number: median thinking tokens per provider and level. If a provider doesn't report `reasoning_tokens`, we fall back to total completion tokens.

## How to run it

```
cd "/Volumes/Powerhouse/Agentic Systems/beamdial"
export HF_TOKEN=your_new_token          # never paste this line anywhere
python3 scripts/probe_effort.py
```

Paste only the table at the end.

## What it costs

Roughly 96 calls averaging a few thousand tokens each, so well under $1 at about $1.20 per million output tokens.
