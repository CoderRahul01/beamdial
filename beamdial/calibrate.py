"""Measure the reasoning-level gap on YOUR prompts, then get the BeamDial settings for them.

Our benchmark says what is true on standard math and code. Your traffic may be harder or different.
Calibrate answers the only question that matters for you: on my prompts, does more thinking buy right answers?

    python -m beamdial calibrate my_prompts.jsonl --levels low,high,max --budget-usd 2

my_prompts.jsonl, one JSON object per line:
    {"prompt": "...", "answer": "42"}            # answer is optional if you pass --check
Custom check (anything you can test: unit tests, JSON schema, a regex):
    --check mymodule:my_check    where  def my_check(text: str, row: dict) -> bool

Everything is saved as it runs (resumable) and the report is printed at the end.
"""
import argparse
import importlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path

from .router import LEVELS


def _last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    depth = 0
    for k in range(j, len(text)):
        depth += text[k] == "{"
        depth -= text[k] == "}"
        if depth == 0:
            return text[j + 1:k]
    return None


def _norm(s):
    s = str(s).strip().strip("$").lower()
    s = re.sub(r"\\(text|mathrm)\{([^}]*)\}", r"\2", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    return re.sub(r"\s+", "", s).rstrip(".")


def default_check(text, row):
    """Compare the last \\boxed{} answer (or the last line) with row["answer"]: exact after normalizing, or numeric."""
    if "answer" not in row:
        raise ValueError("rows need an 'answer' field, or pass --check module:function")
    if not text:
        return False
    pred = _last_boxed(text)
    if pred is None:
        pred = text.strip().splitlines()[-1]
    a, b = _norm(pred), _norm(row["answer"])
    if a == b:
        return True
    try:
        return abs(float(a.replace(",", "")) - float(b.replace(",", ""))) < 1e-6
    except ValueError:
        return False


def load_check(spec):
    if not spec:
        return default_check
    mod, _, fn = spec.partition(":")
    sys.path.insert(0, os.getcwd())
    return getattr(importlib.import_module(mod), fn)


def mcnemar_p(a_only, b_only):
    n, k = a_only + b_only, min(a_only, b_only)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def run(client, model, rows, levels, out_path, max_tokens, price_in, price_out, budget, workers=4, temperature=0.7):
    done = {}
    if out_path.exists():
        for line in out_path.open():
            r = json.loads(line)
            if not r.get("error"):
                done[(r["i"], r["effort"])] = r
    spent = sum(r["prompt_tokens"] * price_in / 1e6 + r["completion_tokens"] * price_out / 1e6 for r in done.values())
    jobs = [(i, lvl) for i in range(len(rows)) for lvl in levels if (i, lvl) not in done]
    lock, state = threading.Lock(), {"spent": spent, "stop": spent >= budget}

    def call(job):
        i, lvl = job
        if state["stop"]:
            return None
        t0 = time.time()
        try:
            r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": rows[i]["prompt"]}],
                                               reasoning_effort=lvl, max_completion_tokens=max_tokens,
                                               temperature=temperature)
        except Exception as ex:  # keep going; failed calls are retried on the next run
            return dict(i=i, effort=lvl, error=str(ex)[:300])
        u = r.usage
        det = getattr(u, "completion_tokens_details", None)
        return dict(i=i, effort=lvl, content=r.choices[0].message.content, finish_reason=r.choices[0].finish_reason,
                    prompt_tokens=u.prompt_tokens or 0, completion_tokens=u.completion_tokens or 0,
                    reasoning_tokens=getattr(det, "reasoning_tokens", None) if det else None,
                    latency_s=round(time.time() - t0, 2))

    with out_path.open("a") as f, ThreadPoolExecutor(workers) as ex:
        for res in ex.map(call, jobs):
            if res is None:
                continue
            with lock:
                f.write(json.dumps(res) + "\n")
                f.flush()
                if res.get("error"):
                    print("error:", res["error"][:120], file=sys.stderr)
                    continue
                done[(res["i"], res["effort"])] = res
                state["spent"] += res["prompt_tokens"] * price_in / 1e6 + res["completion_tokens"] * price_out / 1e6
                if state["spent"] >= budget and not state["stop"]:
                    state["stop"] = True
                    print(f"budget reached (${state['spent']:.2f}); rerun later to resume", file=sys.stderr)
    return done, state["spent"]


def analyze(rows, results, levels, check, tolerance=0.01):
    """Pure function over collected results -> report dict. Separated so it can be tested without an API."""
    ok_ids = [i for i in range(len(rows)) if all((i, l) in results for l in levels)]
    if not ok_ids:
        return {"n": 0}
    correct = {(i, l): bool(check(results[(i, l)]["content"] or "", rows[i])) for i in ok_ids for l in levels}
    tok = {(i, l): results[(i, l)]["completion_tokens"] for i in ok_ids for l in levels}
    think = {(i, l): results[(i, l)].get("reasoning_tokens") or results[(i, l)]["completion_tokens"] for i in ok_ids for l in levels}
    n = len(ok_ids)
    per = {l: dict(accuracy=sum(correct[(i, l)] for i in ok_ids) / n, tokens=sum(tok[(i, l)] for i in ok_ids) / n) for l in levels}
    best = max(levels, key=lambda l: (per[l]["accuracy"], -per[l]["tokens"]))
    cheap = levels[0]
    a_only = sum(1 for i in ok_ids if correct[(i, best)] and not correct[(i, cheap)])
    b_only = sum(1 for i in ok_ids if correct[(i, cheap)] and not correct[(i, best)])
    gap = per[best]["accuracy"] - per[cheap]["accuracy"]
    p = mcnemar_p(a_only, b_only)

    # length-triggered policy: start cheap, re-ask at `best` when cheap thought longer than T
    cands = sorted({think[(i, cheap)] for i in ok_ids})
    policies = []
    for T in [None] + cands:
        acc = toks = 0
        for i in ok_ids:
            if T is not None and think[(i, cheap)] > T:
                acc += correct[(i, best)]
                toks += tok[(i, cheap)] + tok[(i, best)]
            else:
                acc += correct[(i, cheap)]
                toks += tok[(i, cheap)]
        policies.append(dict(threshold=T, accuracy=acc / n, tokens=toks / n))
    policies.append(dict(threshold="best_only", accuracy=per[best]["accuracy"], tokens=per[best]["tokens"]))
    target = per[best]["accuracy"] - tolerance
    good = [q for q in policies if q["accuracy"] >= target]
    pick = min(good, key=lambda q: q["tokens"]) if good else max(policies, key=lambda q: q["accuracy"])

    if all(v["accuracy"] == 0 for v in per.values()):
        verdict = ("Every answer failed the check at every level. That usually means the check or the 'answer' "
                   "field is wrong, not the model. Inspect a few results in the output file.")
    elif best == cheap:
        verdict = f"No level beat {cheap} on your prompts. {cheap} is enough."
    elif p < 0.05 and gap > 0:
        verdict = (f"On your prompts, {best} beats {cheap} by {100 * gap:.1f} points (p = {p:.3f}). More thinking pays here.")
    else:
        verdict = (f"On your prompts, {cheap} is within {100 * max(gap, 0):.1f} points of {best} (p = {p:.2f}, not significant). "
                   f"{cheap} is enough; it uses {100 * per[cheap]['tokens'] / max(per[best]['tokens'], 1):.0f}% of {best}'s tokens.")
    T = pick["threshold"]
    if T is None or (T != "best_only" and T >= cands[-1]):
        setting = f"BeamDial(client, policy='fixed', start='{cheap}')"
    elif T == "best_only":
        setting = f"BeamDial(client, policy='fixed', start='{best}')"
    else:
        setting = (f"BeamDial(client, policy='cascade', start='{cheap}', long_think_tokens={int(T)}, "
                   f"long_think_to='{best}')")
    return dict(n=n, levels=per, best=best, cheap=cheap, gap=gap, p_value=p, cheap_only=b_only, best_only=a_only,
                policy=pick, verdict=verdict, setting=setting)


def print_report(rep, spent):
    if not rep.get("n"):
        print("No prompt has results at every level yet. Rerun to resume.")
        return
    print(f"\nCalibrated on {rep['n']} of your prompts (spent about ${spent:.2f})\n")
    print(f"{'level':8}{'accuracy':>10}{'tokens':>10}")
    for l, v in rep["levels"].items():
        print(f"{l:8}{100 * v['accuracy']:>9.1f}%{v['tokens']:>10,.0f}")
    q = rep["policy"]
    if q["threshold"] is None:
        label = f"best value: always {rep['cheap']}"
    elif q["threshold"] == "best_only":
        label = f"best value: always {rep['best']}"
    else:
        label = f"best value: {rep['cheap']}, re-ask at {rep['best']} if it thinks > {int(q['threshold']):,} tokens"
    print(f"\n{label}: {100 * q['accuracy']:.1f}% at {q['tokens']:,.0f} tokens per prompt")
    print("\n" + rep["verdict"])
    print("Use:\n  " + rep["setting"])
    if rep["n"] < 50:
        print(f"\nNote: {rep['n']} prompts is a small sample. 50 to 200 gives a much firmer answer.")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="beamdial calibrate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompts", help="JSONL with a 'prompt' field (and 'answer' unless --check is given)")
    ap.add_argument("--check", default=None, help="module:function returning True when an answer is acceptable")
    ap.add_argument("--levels", default="low,high,max", help="comma list, cheapest first (default low,high,max)")
    ap.add_argument("--model", default=os.environ.get("BEAM_MODEL", "reflection/Beam-1.0-501B-A23B"))
    ap.add_argument("--base-url", default=os.environ.get("BEAM_BASE_URL", "https://router.huggingface.co/v1"))
    ap.add_argument("--api-key-env", default="HF_TOKEN")
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--budget-usd", type=float, default=2.0)
    ap.add_argument("--price-in", type=float, default=0.40)
    ap.add_argument("--price-out", type=float, default=1.20)
    ap.add_argument("--out", default=None, help="results file (default: <prompts>.calibration.jsonl)")
    args = ap.parse_args(argv)

    levels = args.levels.split(",")
    bad = [l for l in levels if l not in LEVELS]
    if bad or len(levels) < 2:
        ap.error(f"--levels needs at least two of {LEVELS}")
    levels = sorted(levels, key=LEVELS.index)
    rows = [json.loads(l) for l in open(args.prompts) if l.strip()]
    check = load_check(args.check)
    key = os.environ.get(args.api_key_env)
    if not key:
        ap.error(f"set {args.api_key_env} first")
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=key, timeout=900, max_retries=3)
    out = Path(args.out or (args.prompts + ".calibration.jsonl"))
    results, spent = run(client, args.model, rows, levels, out, args.max_tokens, args.price_in, args.price_out, args.budget_usd)
    print_report(analyze(rows, results, levels, check), spent)


if __name__ == "__main__":
    main()
