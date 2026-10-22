"""Graders for BeamDial tasks.

Math: math-verify on the last \\boxed{} answer (falls back to the whole text).
Code: extract the last ```python block, run the official tests in a separate process with a timeout.
Code grading runs model-written code, so run it in a throwaway sandbox (container or VM), not on a laptop you care about.
"""
import os
import re
import subprocess
import sys
import tempfile

from math_verify import parse, verify

CODE_TIMEOUT_S = 15


def load_tasks(root=None) -> dict:
    """All task banks: data/tasks.jsonl (standard set) plus data/tasks_*.jsonl (hard set, long context...)."""
    import json
    from pathlib import Path
    data = Path(root or Path(__file__).resolve().parent.parent) / "data"
    out = {}
    for f in sorted(data.glob("tasks*.jsonl")):
        for line in f.open():
            if line.strip():
                t = json.loads(line)
                out[t["id"]] = t
    return out


def last_boxed(text: str):
    """Return the content of the last \\boxed{...}, handling nested braces."""
    if not text:
        return None
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
        j += 1
    return None


def _clean(a: str) -> str:
    a = str(a).strip().strip("$").strip()
    a = a.replace("\\$", "").replace("\\%", "").replace("%", "")
    a = re.sub(r"\\(text|mbox)\{[^}]*\}", "", a)
    a = a.replace("\\!", "").replace("\\,", "").replace("^\\circ", "").replace("^{\\circ}", "")
    a = a.replace("dfrac", "frac").replace("tfrac", "frac")
    a = re.sub(r"\s+", "", a)  # "15 \\pi" == "15\\pi"
    return a.strip()


def grade_math(content: str, gold: str) -> dict:
    box = last_boxed(content or "")
    if box is None:
        return {"correct": False, "pred": None, "reason": "no_boxed_answer"}
    pred_c, gold_c = _clean(box), _clean(gold)
    if pred_c == gold_c:
        return {"correct": True, "pred": box, "reason": "exact"}
    try:
        # numeric shortcut: 073 vs 73, 6.0 vs 6
        if abs(float(pred_c) - float(gold_c)) < 1e-6:
            return {"correct": True, "pred": box, "reason": "numeric"}
    except ValueError:
        pass
    import threading
    if threading.current_thread() is not threading.main_thread():
        # math-verify's timeout uses signal.alarm, which only works in the main thread. In a worker
        # thread it raises, and an earlier version silently counted that as a wrong answer.
        raise RuntimeError("grade_math must run in the main thread (math-verify uses signal.alarm)")
    try:
        ok = verify(parse("$" + gold_c + "$"), parse("$" + pred_c + "$"))
    except Exception as ex:  # a real parser failure: keep it visible instead of hiding it
        return {"correct": False, "pred": box, "reason": f"grader_error: {type(ex).__name__}"}
    return {"correct": bool(ok), "pred": box, "reason": "math_verify" if ok else "mismatch"}


def extract_code(content: str):
    if not content:
        return None
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", content, flags=re.S)
    if blocks:
        return max(blocks[-3:], key=len) if len(blocks) > 1 else blocks[-1]
    return None


def _run_python(src: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(src)
        try:
            p = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True, text=True,
                               timeout=CODE_TIMEOUT_S, env={"PATH": os.environ.get("PATH", "")})
        except subprocess.TimeoutExpired:
            return False, "timeout"
        return p.returncode == 0, (p.stderr or "")[-300:]


def grade_code(content: str, task: dict) -> dict:
    code = extract_code(content)
    if code is None:
        return {"correct": False, "reason": "no_code_block"}
    if task["source"] == "humaneval":
        # The model may return only the body or the full function; if the entry point is missing, prepend the prompt.
        prog = code if f"def {task['entry_point']}" in code else task["code_prompt"] + code
        prog += "\n\n" + task["test"] + f"\ncheck({task['entry_point']})\n"
    else:
        prog = (task.get("test_setup") or "") + "\n" + code + "\n\n" + "\n".join(task["tests"]) + "\n"
    ok, err = _run_python(prog)
    return {"correct": ok, "reason": "pass" if ok else ("fail: " + err.strip().splitlines()[-1] if err.strip() else "fail")}


def public_test(content: str, task: dict):
    """The check a developer could run without hidden tests: the visible example test.
    MBPP: the one test shown in the prompt. HumanEval: the >>> examples in the docstring.
    Returns True/False, or None when no visible example exists."""
    code = extract_code(content)
    if code is None:
        return False
    if task["source"] == "mbpp":
        prog = (task.get("test_setup") or "") + "\n" + code + "\n\n" + task["tests"][0] + "\n"
    else:
        import doctest
        # closing quotes right after an example would be read as expected output, so blank them out
        doc = task["code_prompt"].replace('"""', "\n").replace("'''", "\n")
        if not doctest.DocTestParser().get_examples(doc):
            return None
        body = code if f"def {task['entry_point']}" in code else task["code_prompt"] + code
        prog = (body + "\n\nimport doctest as _d, sys as _s\n"
                f"_t = _d.DocTestParser().get_doctest({doc!r}, globals(), 'p', None, 0)\n"
                "_r = _d.DocTestRunner(optionflags=_d.ELLIPSIS | _d.NORMALIZE_WHITESPACE)\n"
                "_r.run(_t, out=lambda s: None)\n_s.exit(1 if _r.failures else 0)\n")
    ok, _ = _run_python(prog)
    return ok


def grade(task: dict, content: str) -> dict:
    if task["family"] == "math":
        return grade_math(content, task["answer"])
    return grade_code(content, task)
