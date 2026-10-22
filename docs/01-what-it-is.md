# What BeamDial is

## The problem

Reasoning models think before they answer. That thinking is a stream of hidden tokens. With Beam you control how much it thinks with one parameter:

```python
client.chat.completions.create(model=..., messages=..., reasoning_effort="high")
```

Three things depend on that one value:

- **Cost.** Thinking tokens count as output tokens. On most providers output costs 3 to 4 times more than input.
- **Speed.** More thinking means a longer wait before the first word of the answer.
- **Quality.** On hard problems, more thinking can mean a right answer instead of a wrong one. On easy problems it changes nothing.

Today a developer has no data to choose with. Reflection's docs list the five values but not what each one costs or buys. So most people pick one value for their whole app. That is always wrong for part of their traffic.

## What BeamDial gives people

**1. The measurements (a public dataset).**
410 tasks with known answers, from grade school math to olympiad problems, plus Python coding tasks. Each task is run at all 5 levels. For every run we store: the answer, right or wrong, thinking tokens, total tokens, time, and whether the reply got cut off. Anyone can look up "for tasks like mine, what does `high` buy over `low`?"

**2. The router (a tiny model).**
It reads a prompt and estimates, for each level, the chance Beam gets it right. Then it picks the cheapest level that clears your quality target. It's pure Python, under 1 MB, no GPU, no network call. Your prompt never leaves your machine to get routed.

**3. The wrapper (the library).**
A thin layer around the OpenAI client people already use with Beam:

```python
from openai import OpenAI
from beamdial import BeamDial

beam = BeamDial(OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN))
res = beam.create(messages=[{"role": "user", "content": prompt}])

res.content        # the answer
res.effort         # the level that produced it
res.total_tokens   # tokens paid for, across all attempts
```

It also fixes problems that bite real Beam apps (details in [edge cases](03-edge-cases-and-tradeoffs.md)):

- If you have a check (tests pass, JSON parses, answer matches a format), it climbs one level **only when the check fails**.
- If a reply gets cut off, it retries at the **same** level with more room instead of thinking even longer.
- Tool calls are treated as success, not as an empty answer.
- `res.to_message()` gives you the assistant turn with the `reasoning_content` Beam demands on every follow-up request. Forgetting it breaks the whole chat session with a 400 error.

**4. The Space (a demo anyone can click).**
Paste a prompt, see the level BeamDial picks and the estimated savings. See the charts from the full study. Optionally run it live with your own HF token.

## Who it's for

| Who | What they get |
|---|---|
| Startups building on Beam | Lower bills without guessing, one-line change |
| Agent builders | Safe escalation, tool-call handling, multi-turn fix |
| Hackathon and event devs | A ready starter: "which level should my demo use?" answered with data |
| Researchers | A clean dataset on reasoning effort vs accuracy vs cost |
| Reflection itself | Public, independent evidence about their "3x fewer reasoning tokens" claim |

## How a user actually uses it

1. **Curious:** open the Space, paste a prompt, look at the pick and the charts.
2. **Building:** `pip install git+https://github.com/<you>/beamdial`, wrap the client, done.
3. **Careful:** pass a `verify` function so BeamDial escalates only when the answer fails your own check.
4. **Checking:** `python -m beamdial "your prompt"` prints the per-level estimate locally, no API call.
5. **Researching:** load the dataset from the Hub and filter by task type.

## What it is not

- Not a model. It doesn't change Beam's weights or answers.
- Not a proxy or a hosted service. Nothing sits between the user and their provider.
- Not a general LLM router across different models. It does one job: pick Beam's thinking level.
- Not a guarantee. It estimates from measured data. The docs say exactly where the estimates are weak.

## Why this beats "just use max"

That depends on the full run, so no numbers here yet. What we know from the pilot (14 tasks):

- Beam got 13 of 14 right even at `low`.
- On most tasks the token count barely moved between levels.
- On the hardest tasks `max` sometimes used 5 to 8 times more thinking, and once it thought for 30k tokens and still got it wrong.

So the honest pitch might end up being "BeamDial tells you when `max` is wasted, which is most of the time". The full run decides it.
