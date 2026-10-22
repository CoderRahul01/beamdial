"""Run every task at every Beam reasoning level and log the raw results.

This script only calls the API. It never runs model-written code, so it is safe to run on your laptop.
Grading (which runs code) happens in grade.py inside a sandbox.

Usage:
  export HF_TOKEN=hf_...                 # or BEAM_API_KEY for the Reflection API
  python scripts/collect.py --pilot      # 12 tasks x 5 levels, prints cost and token spread
  python scripts/collect.py --budget-usd 15

Resumable: rerun the same command and finished (task, level, sample) rows are skipped.
"""
import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEVELS = ["low", "medium", "high", "xhigh", "max"]

PROVIDERS = {
    # Hugging Face Inference Providers router. Billed to your HF credits.
    "hf": dict(base_url="https://router.huggingface.co/v1", model="reflection/Beam-1.0-501B-A23B", key_env="HF_TOKEN"),
    # Reflection's own API (OpenAI compatible).
    "reflection": dict(base_url="https://api.reflection.ai/openai/v1", model="Beam-1.0-501B-A23B", key_env="BEAM_API_KEY"),
}


def load_done(path: Path):
    done, spent_in, spent_out = set(), 0, 0
    if path.exists():
        for line in path.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error"):
                continue
            done.add((r["task_id"], r["effort"], r.get("sample", 0)))
            spent_in += r["usage"].get("prompt_tokens", 0)
            spent_out += r["usage"].get("completion_tokens", 0)
    return done, spent_in, spent_out


def pick_pilot(tasks, per_source=2, seed=3):
    """A small pilot: 2 tasks from every source in the task file (14 for the standard set's mix)."""
    rng = random.Random(seed)
    if {"gsm8k", "humaneval"} <= {t["source"] for t in tasks}:  # standard set: keep the original 14-task pilot
        want = {"gsm8k": 2, "math_l3": 2, "math_l5": 2, "aime24": 2, "amc23": 2, "olympiad": 2, "humaneval": 1, "mbpp": 1}
    else:
        want = {s: per_source for s in sorted({t["source"] for t in tasks})}
    out = []
    for s, n in want.items():
        pool = [t for t in tasks if t["source"] == s]
        out += rng.sample(pool, min(len(pool), n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=os.environ.get("BEAM_PROVIDER", "hf"), choices=list(PROVIDERS))
    ap.add_argument("--model", default=None, help="override model id, e.g. reflection/Beam-1.0-501B-A23B:together")
    ap.add_argument("--tasks", default=str(ROOT / "data" / "tasks.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "runs.jsonl"))
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--sources", default="", help="comma list to restrict sources")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--budget-usd", type=float, default=15.0)
    ap.add_argument("--price-in", type=float, default=0.40, help="USD per 1M input tokens (set to your provider's price)")
    ap.add_argument("--price-out", type=float, default=1.20, help="USD per 1M output tokens")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    from openai import OpenAI

    prov = PROVIDERS[args.provider]
    key = os.environ.get(prov["key_env"])
    if not key:
        sys.exit(f"Set {prov['key_env']} first (never paste keys into chat or commit them).")
    model = args.model or os.environ.get("BEAM_MODEL") or prov["model"]
    client = OpenAI(base_url=os.environ.get("BEAM_BASE_URL", prov["base_url"]), api_key=key, timeout=900, max_retries=0)

    tasks = [json.loads(l) for l in open(args.tasks)]
    if args.sources:
        keep = set(args.sources.split(","))
        tasks = [t for t in tasks if t["source"] in keep]
    if args.pilot:
        tasks = pick_pilot(tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    levels = args.levels.split(",")

    out = Path(args.out)
    if args.pilot and args.out == str(ROOT / "data" / "runs.jsonl"):
        out = ROOT / "data" / "pilot_runs.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done, sp_in, sp_out = load_done(out)
    cost = lambda i, o: i / 1e6 * args.price_in + o / 1e6 * args.price_out  # noqa: E731
    state = {"in": sp_in, "out": sp_out, "stop": False, "n": 0, "err": 0}
    lock = threading.Lock()

    jobs = [(t, e, s) for t in tasks for e in levels for s in range(args.samples) if (t["id"], e, s) not in done]
    # interleave levels so an early stop still leaves comparable data across levels
    random.Random(0).shuffle(jobs)
    print(f"model={model} provider={args.provider} jobs={len(jobs)} already_spent=${cost(sp_in, sp_out):.2f} budget=${args.budget_usd}")

    def call(task, effort, sample):
        if state["stop"]:
            return None
        msgs = [{"role": "user", "content": task["prompt"]}]
        for attempt in range(6):
            t0 = time.time()
            try:
                raw = client.chat.completions.with_raw_response.create(
                    model=model, messages=msgs, reasoning_effort=effort,
                    max_completion_tokens=args.max_tokens, temperature=args.temperature,
                )
                r = raw.parse()
                served_by = raw.headers.get("x-inference-provider") or raw.headers.get("x-provider")
                dt = time.time() - t0
                m = r.choices[0].message
                u = r.usage.model_dump() if r.usage else {}
                details = (u.get("completion_tokens_details") or {})
                reasoning = getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None)
                if reasoning is None and getattr(m, "model_extra", None):
                    reasoning = m.model_extra.get("reasoning_content") or m.model_extra.get("reasoning")
                return dict(
                    task_id=task["id"], source=task["source"], family=task["family"], effort=effort, sample=sample,
                    model=model, provider=args.provider, served_by=served_by, content=m.content, reasoning=reasoning,
                    finish_reason=r.choices[0].finish_reason, latency_s=round(dt, 2),
                    usage=dict(prompt_tokens=u.get("prompt_tokens", 0), completion_tokens=u.get("completion_tokens", 0),
                               reasoning_tokens=details.get("reasoning_tokens")),
                    ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
            except Exception as ex:  # retry rate limits and server errors with backoff
                msg = str(ex)
                status = getattr(ex, "status_code", None)
                if status in (400, 401, 403, 404, 422):
                    return dict(task_id=task["id"], effort=effort, sample=sample, error=f"{status}: {msg[:300]}")
                time.sleep(min(60, 2 ** attempt + random.random()))
        return dict(task_id=task["id"], effort=effort, sample=sample, error=f"gave up: {msg[:300]}")

    with out.open("a") as f, ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(call, *j) for j in jobs]
        for fut in as_completed(futs):
            row = fut.result()
            if row is None:
                continue
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                state["n"] += 1
                if row.get("error"):
                    state["err"] += 1
                    print("ERROR", row["task_id"], row["effort"], row["error"][:160])
                else:
                    state["in"] += row["usage"]["prompt_tokens"]
                    state["out"] += row["usage"]["completion_tokens"]
                spent = cost(state["in"], state["out"])
                if spent >= args.budget_usd and not state["stop"]:
                    state["stop"] = True
                    print(f"Budget reached (${spent:.2f}). Stopping new calls; rerun later to resume.")
                if state["n"] % 20 == 0:
                    print(f"{state['n']}/{len(jobs)} done, errors={state['err']}, est spend=${spent:.2f}")

    print(f"Finished. rows={state['n']} errors={state['err']} est total spend=${cost(state['in'], state['out']):.2f} -> {out}")
    summarize(out)


def summarize(path: Path):
    rows = [json.loads(l) for l in path.open() if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    if not rows:
        return
    print("\nlevel    n   median_reasoning_tok  median_completion_tok  truncated  median_latency_s")
    for e in LEVELS:
        rs = [r for r in rows if r["effort"] == e]
        if not rs:
            continue
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None  # noqa: E731
        rt = [r["usage"]["reasoning_tokens"] for r in rs if r["usage"].get("reasoning_tokens") is not None]
        ct = [r["usage"]["completion_tokens"] for r in rs]
        tr = sum(r["finish_reason"] == "length" for r in rs)
        print(f"{e:7} {len(rs):4}   {str(med(rt)):>20}  {med(ct):>21}  {tr:>9}  {med([r['latency_s'] for r in rs]):>16}")
    print("\nIf the token medians barely change across levels, the provider is ignoring reasoning_effort."
          " Try --provider reflection or pin a provider with --model reflection/Beam-1.0-501B-A23B:<provider>.")


if __name__ == "__main__":
    main()
