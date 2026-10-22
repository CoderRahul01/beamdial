"""Check whether reasoning_effort actually changes Beam's thinking, per provider.

Runs 4 harder pilot prompts at low / medium / max, 2 samples each, on every provider you list.
Cost is well under $1. Prints median reasoning tokens per provider and level.

  export HF_TOKEN=...            # new token
  export BEAM_API_KEY=...        # optional, adds Reflection's own API
  python3 scripts/probe_effort.py
"""
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
TASK_IDS = ["aime24-73", "math-precalculus-1081", "olympiad-2759", "olympiad-2745"]
LEVELS = ["low", "medium", "max"]
SAMPLES = 2
HF_PROVIDERS = ["together", "fireworks-ai", "baseten", "nebius"]


def targets():
    out = []
    if os.environ.get("HF_TOKEN"):
        c = OpenAI(base_url="https://router.huggingface.co/v1", api_key=os.environ["HF_TOKEN"], timeout=900, max_retries=2)
        for p in HF_PROVIDERS:
            out.append((f"hf:{p}", c, f"reflection/Beam-1.0-501B-A23B:{p}"))
    if os.environ.get("BEAM_API_KEY"):
        c = OpenAI(base_url="https://api.reflection.ai/openai/v1", api_key=os.environ["BEAM_API_KEY"], timeout=900, max_retries=2)
        out.append(("reflection", c, "Beam-1.0-501B-A23B"))
    if not out:
        sys.exit("Set HF_TOKEN and/or BEAM_API_KEY first.")
    return out


def main():
    sys.path.insert(0, str(ROOT / "scripts"))
    from graders import load_tasks
    tasks = load_tasks(ROOT)
    jobs = [(name, c, m, tid, lvl, s) for name, c, m in targets() for tid in TASK_IDS for lvl in LEVELS for s in range(SAMPLES)]
    dead = set()
    out_path = ROOT / "data" / "probe_runs.jsonl"

    def run(job):
        name, c, model, tid, lvl, s = job
        if name in dead:
            return None
        t0 = time.time()
        try:
            raw = c.chat.completions.with_raw_response.create(
                model=model, messages=[{"role": "user", "content": tasks[tid]["prompt"]}],
                reasoning_effort=lvl, max_completion_tokens=32000, temperature=0.7)
            r = raw.parse()
            u = r.usage.model_dump()
            served = raw.headers.get("x-inference-provider") or raw.headers.get("x-provider")
            return dict(target=name, served_by=served, task_id=tid, effort=lvl, sample=s, content=r.choices[0].message.content,
                        finish_reason=r.choices[0].finish_reason, latency_s=round(time.time() - t0, 2),
                        usage=dict(prompt_tokens=u.get("prompt_tokens", 0), completion_tokens=u.get("completion_tokens", 0),
                                   reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens")))
        except Exception as ex:
            status = getattr(ex, "status_code", None)
            if status in (400, 401, 403, 404, 422):
                dead.add(name)
            return dict(target=name, task_id=tid, effort=lvl, sample=s, error=f"{status}: {str(ex)[:200]}")

    with ThreadPoolExecutor(6) as ex, out_path.open("a") as f:
        rows = []
        for r in ex.map(run, jobs):
            if r is None:
                continue
            f.write(json.dumps(r) + "\n")
            rows.append(r)

    errs = {}
    for r in rows:
        if r.get("error"):
            errs.setdefault(r["target"], r["error"])
    for k, v in errs.items():
        print(f"{k}: ERROR {v}")
    print(f"\n{'target':18} {'served_by':14}" + "".join(f"{l:>10}" for l in LEVELS) + "   (median reasoning tokens)")
    for name in dict.fromkeys(r["target"] for r in rows):
        ok = [r for r in rows if r["target"] == name and not r.get("error")]
        if not ok:
            continue
        served = next((r["served_by"] for r in ok if r.get("served_by")), "-")
        meds = [statistics.median([(r["usage"]["reasoning_tokens"] or r["usage"]["completion_tokens"]) for r in ok if r["effort"] == l] or [0]) for l in LEVELS]
        print(f"{name:18} {str(served):14}" + "".join(f"{m:>10,.0f}" for m in meds))
    print(f"\nraw rows -> {out_path}")


if __name__ == "__main__":
    main()
