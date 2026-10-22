# Learning path

This project touches six areas of AI engineering. For each one: what it is, where it shows up in BeamDial, what to read, and one exercise you can do in this repo.

It also maps onto your 12-stage module. BeamDial covers parts of Stage 4 (verified data), Stage 8 (inference cost), Stage 10 (evals), Stage 11 (paper to code) and Stage 12 (public portfolio).

Search the titles below. The papers are all free on arXiv.

---

## 1. Reasoning models and test-time compute

**What it is.** Older models answer right away. Reasoning models "think" first, writing hidden tokens before the answer. Spending more compute at answer time (test-time compute) can beat training a bigger model, but only on problems hard enough to need it. Too much thinking on easy problems is called **overthinking**.

**In BeamDial.** `reasoning_effort` is a test-time compute dial. Our whole dataset measures how much it buys.

**Read.**
- "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters" (Snell et al., 2024). The core idea: the best amount of thinking depends on how hard the prompt is.
- "Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs" (Chen et al., 2024). Measures wasted thinking on easy problems.
- "s1: Simple test-time scaling" (Muennighoff et al., 2025). Shows how a thinking budget can be forced up or down.
- "DeepSeek-R1" (DeepSeek-AI, 2025). How open reasoning models are trained.

**Exercise.** After the full run, open `results/levels.png`. Find the task group where `max` adds the most accuracy per extra token. Write one sentence on why.

---

## 2. LLM routing and cascades

**What it is.** A **router** sends each request to the cheapest option likely to handle it. A **cascade** tries cheap first and escalates on failure. Most published routers pick between different models. BeamDial picks between thinking levels of one model, which is less studied.

**In BeamDial.** `router.py` is a router. `client.py` with `verify` is a cascade. `analyze.py` compares both.

**Read.**
- "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance" (Chen, Zaharia, Zou, 2023). The original cascade paper.
- "RouteLLM: Learning to Route LLMs with Preference Data" (Ong et al., 2024). Learned routers between a strong and a weak model.
- "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (Wang et al., 2022). The idea behind our "agreement cascade": if two attempts agree, trust the answer.

**Exercise.** In `analyze.py`, add a policy "start at `medium`, escalate on failed check, but never above `high`". Rerun the analysis and see where it lands on the chart.

---

## 3. Evaluation and grading

**What it is.** A benchmark is only as good as its grader. Common traps: grading text by exact match (fails on "0.5" vs "1/2"), leaking hidden tests into the method, random variation mistaken for real differences, and training-data contamination.

**In BeamDial.** `graders.py` (math-verify, sandboxed code tests), the gold-answer check, the "visible test only" rule for escalation, and the note on one sample per cell.

**Read.**
- "Let's Verify Step by Step" (Lightman et al., 2023). Also where the MATH-500 split comes from.
- The Hugging Face `math-verify` README. Why math answer checking is hard.
- "Evaluating Large Language Models Trained on Code" (Chen et al., 2021). Introduces HumanEval and pass@k.

**Exercise.** Find 3 graded rows where `grade_reason` is `mismatch` and read Beam's answer. Is the model wrong, or is the grader? This is how real benchmark bugs get found.

---

## 4. Small, honest machine learning

**What it is.** Logistic regression, the hashing trick, cross-validation, regularization and calibration. These are the basics that make a small model trustworthy.

**In BeamDial.** `features.py` (hashing), `train_router.py` (one logistic model per level, 5-fold CV, leave-one-group-out).

**Key ideas.**
- **Out-of-fold predictions:** never score a model on data it trained on.
- **Leave-one-group-out:** test on a whole kind of data the model never saw. It's closer to real users.
- **Calibration:** when the model says 80%, is it right 80% of the time? Our target slider only makes sense if it is.

**Read.**
- scikit-learn user guide: "Cross-validation" and "Probability calibration".
- "On Calibration of Modern Neural Networks" (Guo et al., 2017).
- "Feature Hashing for Large Scale Multitask Learning" (Weinberger et al., 2009).

**Exercise.** Add a calibration check to `analyze.py`: bucket the out-of-fold probabilities (0 to 0.2, 0.2 to 0.4 ...) and print the real accuracy in each bucket.

---

## 5. Inference cost and serving

**What it is.** How models are priced and served. Input vs output token prices, why reasoning tokens are billed as output, latency, and why the same model behaves differently across providers.

**In BeamDial.** The cost model in `analyze.py`, `served_by` in `collect.py`, and the whole probe.

**Read.**
- Hugging Face "Inference Providers" docs (the router, provider pinning, billing).
- Reflection docs: "Reasoning" and "Rate limits" pages.
- The Beam launch blog on Hugging Face: the KV-cache and MoE sections explain why a 501B model runs like a 23B one.

**Exercise.** Using `summary.json`, work out the monthly bill for an app doing 100k requests a day at fixed `max` vs with BeamDial. That one number is your best sales line.

---

## 6. Shipping an open-source tool

**What it is.** Packaging, a README that sells in 10 seconds, tests in CI, a demo people can click, a dataset card and a model card.

**In BeamDial.** `pyproject.toml`, `tests/`, `.github/workflows/`, `space/`, the docs you're reading.

**Read.**
- Hugging Face docs: "Dataset cards" and "Model cards".
- Python Packaging User Guide: "Packaging Python Projects".

**Exercise.** Publish the dataset with a card that answers: what's in it, how it was made, known limits, license.

---

## What to learn next (after the hackathon)

1. **Async and streaming in Python** (asyncio, httpx). Needed for `AsyncBeamDial` and streaming support.
2. **Agent evals** (tool calling, multi-step tasks). The most valuable next dataset.
3. **Small-model fine-tuning with LoRA** (TRL, PEFT). Train a stronger router on the same data, then compare against logistic regression.
4. **Process reward models.** Judge the thinking, not just the final answer.
