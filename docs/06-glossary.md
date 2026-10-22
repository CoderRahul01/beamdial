# Glossary

**Beam.** Reflection's open model: 501B parameters in total, 23B used per token, 1M token context, Apache 2.0 license.

**MoE (mixture of experts).** A model built from many small sub-networks ("experts"). Each token uses only a few of them, so a 501B model runs about as fast as a 23B one.

**Reasoning tokens / thinking tokens.** Hidden text the model writes before its answer. It's returned as `reasoning_content` and billed as output tokens.

**`reasoning_effort`.** Beam's dial for how much to think: `low`, `medium`, `high`, `xhigh`, `max`.

**Completion tokens.** Everything the model writes: reasoning + visible answer. This is what you pay output prices for.

**`max_completion_tokens`.** The cap on completion tokens for one call. If the model hits it, the reply is cut off (`finish_reason = "length"`).

**`finish_reason`.** Why the model stopped. `stop` = finished normally, `length` = hit the cap, `tool_calls` = wants to call a tool.

**Inference provider.** A company that runs the model on its GPUs and sells API access (Together, Fireworks, Baseten, Nebius...). Hugging Face's router forwards your request to one of them.

**Provider pinning.** Forcing a specific provider on the HF router, for example `reflection/Beam-1.0-501B-A23B:together`.

**Router.** Something that decides, per request, which option to use. Here: which reasoning level.

**Cascade / escalation.** Try a cheap option first, move to a stronger one only if the result fails a check.

**Verify / check.** A function you provide that says whether an answer is acceptable: tests pass, JSON parses, a regex matches.

**Oracle.** A strategy that magically knows the cheapest level that works for each task. Impossible in practice, used to show the best possible result.

**Pareto frontier.** On a cost vs accuracy chart, the strategies where you can't get more accuracy without paying more. Anything below and to the right of it is wasteful.

**Ground truth / gold answer.** The known correct answer, used for grading.

**Hidden test vs visible test.** Hidden tests are the benchmark's full test suite, used only to score. A visible test is an example a developer could see and run themselves.

**Logistic regression.** A simple model that turns weighted features into a probability between 0 and 1.

**Hashing trick.** Turning words into numbers by hashing them into a fixed number of slots, with no vocabulary file.

**Cross-validation (cv5).** Split data into 5 parts. Train on 4, predict the 5th, rotate. Every prediction comes from a model that didn't see that example.

**Leave-one-group-out (loso).** Hold out a whole kind of task (say all competition math), train on the rest, predict the held-out kind. A tougher, more realistic test.

**Calibration.** Whether predicted probabilities match reality. 80% predictions should be right about 80% of the time.

**Temperature.** Randomness in the model's output. 0.7 means the same prompt can give different answers on different runs.

**Sandbox.** An isolated environment for running untrusted code, so model-written code can't touch your real machine.

**Space.** A free web app hosted on Hugging Face.

**Dataset card / model card.** The README for a dataset or model on the Hub: what it is, how it was made, limits, license.
