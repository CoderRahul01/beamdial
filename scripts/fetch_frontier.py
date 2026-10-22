"""Download the frontier math sets from MathArena on Hugging Face -> data/tasks_frontier.jsonl.

These are much harder than the v0.1 set, so the reasoning level has room to matter:
  - AIME 2026 (30), HMMT February 2026 (33): recent competitions with integer / closed-form answers
  - Apex 2025 (12) and Apex Shortlist (47): problems picked to be hard for frontier models

Contamination note: all of these predate Beam's knowledge cutoff (June 30, 2026), so Beam may have seen them.
That's why the long-context set (scripts/build_longctx.py) exists: it's generated fresh and can't be memorized.

Run on your Mac (needs internet to huggingface.co):
  python3 scripts/fetch_frontier.py
"""
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "tasks_frontier.jsonl"
MATH_INSTR = "Solve the following problem. Put only the final answer inside \\boxed{}.\n\n"

SETS = [  # (hf dataset id, our source name, difficulty hint)
    ("MathArena/aime_2026", "aime26", 5),
    ("MathArena/hmmt_feb_2026", "hmmt26", 5),
    ("MathArena/apex_2025", "apex", 5),
    ("MathArena/apex-shortlist", "apex_short", 5),
]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "beamdial"})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def rows_for(dataset):
    q = urllib.parse.quote(dataset, safe="")
    splits = get(f"https://datasets-server.huggingface.co/splits?dataset={q}")["splits"]
    split = next((s for s in splits if s["split"] in ("train", "test")), splits[0])
    out, off = [], 0
    while True:
        page = get(f"https://datasets-server.huggingface.co/rows?dataset={q}&config={split['config']}"
                   f"&split={split['split']}&offset={off}&length=100")
        rows = [r["row"] for r in page["rows"]]
        out += rows
        off += len(rows)
        if not rows or off >= page.get("num_rows_total", off):
            return out


def pick(row, *names):
    for n in names:
        for k, v in row.items():
            if k.lower() == n and v not in (None, ""):
                return v
    return None


def main():
    tasks, skipped = [], []
    for ds, source, hint in SETS:
        try:
            rows = rows_for(ds)
        except Exception as ex:  # one missing set shouldn't block the others
            print(f"SKIP {ds}: {ex}")
            continue
        if rows:
            print(f"{ds}: {len(rows)} rows, columns: {sorted(rows[0])}")
        for i, r in enumerate(rows):
            problem = pick(r, "problem", "question", "statement")
            answer = pick(r, "answer", "final_answer", "gold_answer", "solution_answer")
            idx = pick(r, "problem_idx", "id", "idx") or i + 1
            if problem is None or answer is None or isinstance(answer, (list, dict)):
                skipped.append((ds, idx))
                continue
            tasks.append(dict(id=f"{source}-{idx}", source=source, family="math", difficulty_hint=hint,
                              prompt=MATH_INSTR + str(problem), answer=str(answer).strip().strip("$")))
    if not tasks:
        sys.exit("Nothing downloaded. Check your internet connection to huggingface.co and paste the error here.")
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate ids"
    OUT.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks))
    print(f"\n{len(tasks)} tasks -> {OUT}  (skipped {len(skipped)} without a single final answer)")


if __name__ == "__main__":
    main()
