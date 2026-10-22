"""Grade raw runs. Runs model-written code, so use a sandbox (container/VM).

python scripts/grade.py data/runs.jsonl  ->  data/graded.jsonl
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graders import grade, last_boxed, load_tasks, public_test  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "runs.jsonl")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "data" / "graded.jsonl")
    tasks = load_tasks(ROOT)
    rows = [json.loads(l) for l in src.open() if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    # last write wins if a (task, effort, sample) was collected twice
    uniq = {(r["task_id"], r["effort"], r.get("sample", 0)): r for r in rows}
    rows = list(uniq.values())

    def work(r):
        t = tasks[r["task_id"]]
        g = grade(t, r.get("content") or "")
        r = dict(r)
        r["correct"] = bool(g["correct"])
        r["grade_reason"] = g["reason"]
        if t["family"] == "math":
            r["pred"] = last_boxed(r.get("content") or "")
            r["public_pass"] = None
        else:
            r["pred"] = None
            r["public_pass"] = public_test(r.get("content") or "", t)
        return r

    # Math in the main thread (math-verify needs it), code in threads (each test runs in its own process).
    math_rows = [r for r in rows if tasks[r["task_id"]]["family"] == "math"]
    code_rows = [r for r in rows if tasks[r["task_id"]]["family"] != "math"]
    out = [work(r) for r in math_rows]
    with ThreadPoolExecutor(8) as ex:
        out += list(ex.map(work, code_rows))
    out.sort(key=lambda r: (r["task_id"], r["effort"], r.get("sample", 0)))
    dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    n = len(out)
    errs = sum(str(r.get("grade_reason", "")).startswith("grader_error") for r in out)
    print(f"graded {n} rows -> {dst}; overall accuracy {sum(r['correct'] for r in out) / max(n, 1):.3f}; grader errors {errs}")


if __name__ == "__main__":
    main()
