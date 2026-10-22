# How it works

There are two halves:

- **The research pipeline** (`scripts/`). It runs once. It turns tasks into measurements, measurements into a trained router, and the router into charts and a report.
- **The product** (`beamdial/` and `space_static/`). This is what users install or click. It only reads the trained router file.

```
tasks.jsonl ──> collect.py ──> runs.jsonl ──> grade.py ──> graded.jsonl ──┬──> train_router.py ──> router_weights.json ──> beamdial library
  (410 tasks)    (calls Beam     (raw replies)  (checks         (right/wrong)  │                                              + Space
                  at 5 levels)                   answers)                      └──> analyze.py ──> report.md, charts, summary.json
```

Why it's split like this:

- `collect.py` only talks to the API, so it's safe to run anywhere.
- `grade.py` **runs code that Beam wrote**, so run it in a sandbox (a container or a throwaway VM), not on a machine you care about.
- Everything after collection is offline. We can re-grade, retrain and re-chart for free, as many times as we like.

Below is every file, what it does, and the lines that matter. Line numbers drift as code changes, so the code is quoted instead.

---

## scripts/build_tasks.py

**Job:** build `data/tasks.jsonl`, the 410 tasks.

```python
PLAN = {"gsm8k": 40, "math_l1": 20, ..., "aime24": 30, "olympiad": 40, "humaneval": 50, "mbpp": 50}
```
How many tasks we take from each source. The mix is deliberate: easy (GSM8K), graded difficulty (MATH levels 1 to 5), very hard (AMC, AIME, Olympiad), and code (HumanEval, MBPP). A router can only learn "when does more thinking help" if the data has both easy and hard tasks.

```python
SEED = 7
rng = random.Random(SEED)
```
Fixed random seed, so anyone who reruns this gets the exact same 410 tasks. Reproducibility is what makes a benchmark trustworthy.

```python
MATH_INSTR = "Solve the following problem. Put only the final answer inside \\boxed{}."
```
Every math prompt asks for the answer in `\boxed{}`. That gives the grader one clear place to look.

```python
def simple_answer(ans): ... if "\\text" in a or "," in a ... return False
```
Skips MATH problems whose answers are text or multi-part (like "(2, 3)" or "\text{east}"). Those are hard to grade automatically, and a grading mistake would look like a model mistake.

```python
mbpp = [r for r in jl("mbpp.jsonl") if 11 <= r["task_id"] <= 510]
```
Uses only MBPP's official test split (ids 11 to 510).

```python
assert len(ids) == len(set(ids)), "duplicate task ids"
```
A safety check. Two tasks with the same id would silently merge results later.

---

## scripts/graders.py

**Job:** decide right or wrong for one reply.

```python
def last_boxed(text): ...
```
Finds the **last** `\boxed{...}` in the reply and handles nested braces like `\boxed{\frac{1}{2}}`. Last, because models sometimes box intermediate steps.

```python
def grade_math(content, gold):
    if pred_c == gold_c: exact
    if abs(float(pred_c) - float(gold_c)) < 1e-6: numeric     # "073" == "73", "6.0" == "6"
    ok = verify(parse(gold), parse(pred))                     # math-verify: 1/2 == 0.5 == \frac{1}{2}
```
Three checks from cheap to smart. `math-verify` is Hugging Face's library for checking that two math expressions are equal.

```python
def grade_code(content, task):
    code = extract_code(content)                 # last ```python block
    prog = code + test + f"check({entry_point})" # HumanEval
    ok, err = _run_python(prog)                  # separate process, 15 s timeout
```
Code is graded by running the official hidden tests. `_run_python` uses a fresh temp folder, `python -I` (isolated mode) and an empty environment, and kills it after 15 seconds so an infinite loop can't hang the run.

```python
def public_test(content, task):
```
This one matters for the product. It runs only the test a developer could **see**: the example in the MBPP prompt, or the `>>>` examples in a HumanEval docstring. When we simulate "escalate when the check fails", we use this visible test, never the hidden ones. Using hidden tests would be cheating, because real users don't have them.

The graders were tested: every gold answer and every reference solution passes, wrong answers fail, and an infinite loop times out.

---

## scripts/collect.py

**Job:** call Beam for every (task, level) pair and save the raw reply.

```python
PROVIDERS = {"hf": dict(base_url="https://router.huggingface.co/v1", model="reflection/Beam-1.0-501B-A23B", key_env="HF_TOKEN"), ...}
```
Two ways to reach Beam: Hugging Face's router (billed to your HF credits) or Reflection's own API.

```python
raw = client.chat.completions.with_raw_response.create(model=model, messages=msgs, reasoning_effort=effort,
                                                       max_completion_tokens=args.max_tokens, temperature=args.temperature)
served_by = raw.headers.get("x-inference-provider")
```
The actual call. `with_raw_response` lets us read the HTTP headers too, so we record **which provider really served the request**. Hugging Face picks one automatically unless you pin it.

```python
usage=dict(prompt_tokens=..., completion_tokens=..., reasoning_tokens=details.get("reasoning_tokens"))
```
The three numbers the whole project is about. `reasoning_tokens` sits inside `completion_tokens`, so completion = reasoning + visible answer.

```python
done, spent_in, spent_out = load_done(path)
jobs = [... if (t["id"], e, s) not in done]
```
**Resumable.** Rerun the same command after a crash or a budget stop and it skips everything already saved. Every row is written and flushed the moment it arrives.

```python
random.Random(0).shuffle(jobs)
```
Shuffles the order, so if the budget runs out halfway, every level has roughly the same coverage. Without this you could end up with all `low` runs done and no `max` runs.

```python
if spent >= args.budget_usd: state["stop"] = True
```
**The budget guard.** It estimates spend from token counts and stops sending new calls once it reaches `--budget-usd`. Calls already in flight finish, so the real total can go a little over.

```python
if status in (400, 401, 403, 404, 422): return error   # our mistake, don't retry
time.sleep(min(60, 2 ** attempt + random.random()))    # rate limit / server error: back off and retry
```
Retries server hiccups with growing waits (exponential backoff with jitter), but doesn't retry errors that will never succeed, like a bad key.

---

## scripts/grade.py

**Job:** run the graders over every saved reply, in a sandbox.

```python
uniq = {(r["task_id"], r["effort"], r.get("sample", 0)): r for r in rows}
```
If a pair was collected twice, the last one wins. No double counting.

```python
r["public_pass"] = public_test(...)
```
Stores the visible-test result next to the real grade, so the analysis can simulate "escalate on failed check" honestly.

---

## beamdial/features.py

**Job:** turn a prompt into numbers the router can use.

```python
DIM = 1 << 14   # 16,384 slots
idx = crc32(word) % DIM
```
This is the **hashing trick**. Instead of building a vocabulary, every word and word pair is hashed into one of 16,384 slots. Benefits: no vocabulary file, works on words it has never seen, tiny and fast. Cost: two different words can land in the same slot sometimes. That's fine for a router.

```python
grams = toks + [a + " " + b for a, b in zip(toks, toks[1:])]
```
Single words and word pairs. Pairs catch phrases like "how many" or "prove that".

```python
dense = [len(text), digit share, LaTeX command count, has code fence, has "def ", max number, ...] + keyword flags
```
A few hand-picked signals. Long LaTeX-heavy prompts with big numbers tend to be harder. Code prompts behave differently from math.

Pure Python with `zlib.crc32`, so the same prompt gives the same features on any machine. No numpy needed at runtime.

---

## scripts/train_router.py

**Job:** learn, for each level, P(Beam is right | prompt).

```python
for l in LEVELS:
    m = LogisticRegression(C=0.5, solver="liblinear").fit(X, y_at_level_l)
```
One small **logistic regression** per level. Logistic regression outputs a probability between 0 and 1, trains in seconds, and its weights can be saved as a plain JSON file. `C=0.5` adds regularization so it doesn't memorize 410 examples.

```python
if yb.min() == yb.max(): return None, log(p / (1 - p))
```
If Beam got **every** task right at some level, there's nothing to learn, so we store a constant probability instead of crashing.

```python
skf = StratifiedKFold(n_splits=5) ...           # cv5
for g in groups: train on others, predict g     # loso
```
Honest scoring. Every task's prediction comes from a model that **never saw that task**. Two versions:
- `cv5`: 5-fold cross-validation. Each fold has a mix of all sources.
- `loso` (leave one source group out): the router is tested on a whole **kind** of task it never trained on, for example all competition math. This is the realistic case of a user with a new kind of prompt.

```python
weights = dict(coef=..., intercept=..., tokens=table, default_target=0.85)
```
The shipped file. `tokens` is the median token count per level for math and code, used to show estimated savings.

---

## beamdial/router.py

**Job:** load the weights and pick a level. This runs on the user's machine.

```python
z = intercept[lvl] + sum(v * coef[lvl][i] for i, v in features)
p = sigmoid(z)
```
A logistic regression prediction written by hand, so the library needs nothing installed.

```python
best = max(best, out[lvl]); out[lvl] = best
```
**Monotone smoothing.** Thinking more rarely makes things worse on average, so a higher level never gets a lower estimate than a cheaper one. This stops odd picks like "use `high`" when `medium` was estimated better.

```python
for lvl in levels:
    if pr[lvl] >= target: pick = lvl; break
```
Picks the **cheapest** level whose estimate clears the target. Target 0.85 means "I want roughly an 85% chance of a right answer". Higher target, safer and more tokens.

```python
def family(prompt): -> "code" | "math" | "other"
```
Out-of-domain check. The router only learned math and Python. For anything else (emails, summaries) its estimate is a guess, so the wrapper uses a fixed fallback level instead.

---

## beamdial/client.py

**Job:** the wrapper users call. Full edge-case list in [edge cases](03-edge-cases-and-tradeoffs.md).

```python
routed = effort or self.pick(messages)
```
Uses the router's pick unless the caller forces a level.

```python
if ch.finish_reason == "length": outcome = "truncated"
elif has_tools: outcome = "tool_call"
elif not content: outcome = "empty"
else: outcome = "ok" or "verify_failed"
```
Every reply gets one clear label. The next step depends on it.

```python
if outcome == "truncated":
    kwargs["max_completion_tokens"] = min(budget * 2, self.max_tokens_cap); continue
```
Cut off? Same level, double the room. Climbing a level here would think longer and get cut off again.

```python
cur = LEVELS[i + 1]
```
Only `verify_failed` or `empty` climbs a level, and only up to `max_effort`, at most `max_attempts` calls.

```python
def to_message(self): ... "reasoning_content": self.reasoning or REASONING_PLACEHOLDER
```
Beam rejects any assistant message without `reasoning_content`. Some providers drop it on tool-call turns. We always fill it, the same fix Reflection documents for the Hermes agent.

---

## scripts/analyze.py

**Job:** replay strategies on the real data and write the report.

The key idea: because **every task was run at every level**, we can replay any strategy without new API calls. "What if you always used `high`?" Look up the `high` result for every task. "What if you start at `low` and climb on a failed check?" Walk up the levels using the saved results.

Strategies compared:
- **fixed: low ... max**: one level for everything. The baseline.
- **oracle**: the cheapest level that was right, per task. Impossible in practice (it needs the answer key) but shows the ceiling.
- **cascade**: start low, climb when the visible check fails.
- **agreement cascade**: for math, run two levels and stop when they agree. Costs at least 2 calls but needs no checker.
- **router @ target**: BeamDial's pick, with and without escalation, scored out-of-fold.

Outputs: `report.md` (tables), `pareto.png` (accuracy vs tokens, the main chart), `levels.png` (tokens and accuracy per level per task group), `summary.json` (every number, used by the demo).

---

## scripts/probe_effort.py

Explained line by line in [the probe](04-the-probe.md).

---

## space_static/

The demo is a static page. `router.js` is a line-for-line port of `features.py` and `router.py`, so the router runs in the visitor's browser and the prompt is never sent anywhere. `scripts/check_js_parity.py` runs both versions on every task in the dataset and fails if any pick differs or any probability differs by more than 1e-9.

---

## tests/

- `test_beamdial.py`: the wrapper and router with a fake Beam client: router picks, out-of-domain fallback, escalation order, attempt cap, tool calls, truncation, `to_message`, cost, the long-think option.
- `test_graders.py`: equivalent math answers must grade as correct, and the grader must refuse to run where math-verify silently fails.
- `test_calibrate.py`: the calibrate analysis, including the case where a user's check is broken.
- `mock_server.py`: a fake OpenAI-compatible server, used to run the whole pipeline end to end before spending anything.

Run them with `python -m pytest -q`.
