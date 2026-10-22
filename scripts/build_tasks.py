"""Build the BeamDial task bank: auto-gradable math and code tasks with a wide difficulty spread.

Every task has: id, source, family (math|code), difficulty_hint, prompt, and grading info.
Selection is seeded so the bank is reproducible.
"""
import ast
import json
import random
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent.parent / "data" / "tasks.jsonl"
SEED = 7

MATH_INSTR = (
    "Solve the following problem. Put only the final answer inside \\boxed{}.\n\n"
)
CODE_INSTR = (
    "Write a correct Python solution. Return the complete code in a single ```python code block "
    "and nothing else after it.\n\n"
)

# How many tasks to take from each source. difficulty_hint is a rough prior from the source, 1 easy .. 5 hardest.
PLAN = {
    "gsm8k": 40,
    "math_l1": 20,
    "math_l2": 20,
    "math_l3": 30,
    "math_l4": 40,
    "math_l5": 50,
    "amc23": 40,
    "aime24": 30,
    "olympiad": 40,
    "humaneval": 50,
    "mbpp": 50,
}


def jl(name):
    return [json.loads(line) for line in open(RAW / name) if line.strip()]


def take(rng, items, n):
    items = list(items)
    rng.shuffle(items)
    return items[:n]


def simple_answer(ans: str) -> bool:
    """Keep MATH answers that a parser can check reliably (skip text answers and multi-part answers)."""
    a = str(ans).strip()
    if not a or len(a) > 40:
        return False
    if "\\text" in a or "\\mbox" in a or "," in a or "\\cup" in a or "=" in a:
        return False
    return True


def main():
    rng = random.Random(SEED)
    tasks = []

    for r in take(rng, jl("gsm8k.jsonl"), PLAN["gsm8k"]):
        ans = r["answer"].split("####")[-1].strip().replace(",", "")
        tasks.append(dict(id=f"gsm8k-{r['idx']}", source="gsm8k", family="math", difficulty_hint=1,
                          prompt=MATH_INSTR + r["question"], answer=ans))

    math_rows = [r for r in jl("math.jsonl") if simple_answer(r["answer"])]
    for lvl in range(1, 6):
        pool = [r for r in math_rows if r["level"] == f"Level {lvl}"]
        for r in take(rng, pool, PLAN[f"math_l{lvl}"]):
            tasks.append(dict(id="math-" + r["unique_id"].replace("test/", "").replace(".json", "").replace("/", "-"),
                              source=f"math_l{lvl}", family="math", difficulty_hint=min(lvl, 4),
                              prompt=MATH_INSTR + r["problem"], answer=str(r["answer"]), subject=r["subject"]))

    for r in take(rng, jl("amc23.jsonl"), PLAN["amc23"]):
        a = r["answer"]
        a = str(int(a)) if float(a).is_integer() else str(a)
        tasks.append(dict(id=f"amc23-{r['id']}", source="amc23", family="math", difficulty_hint=4,
                          prompt=MATH_INSTR + r["question"], answer=a))

    for r in take(rng, jl("aime24.jsonl"), PLAN["aime24"]):
        tasks.append(dict(id=f"aime24-{r['id']}", source="aime24", family="math", difficulty_hint=5,
                          prompt=MATH_INSTR + r["problem"], answer=str(r["answer"])))

    oly = [r for r in jl("olympiad.jsonl")
           if r.get("answer_type") == "Numerical" and not r.get("is_multiple_answer")
           and str(r.get("unit")) in ("None", "", "null")]
    for r in take(rng, oly, PLAN["olympiad"]):
        fa = r["final_answer"]
        fa = fa[0] if isinstance(fa, list) else ast.literal_eval(fa)[0] if str(fa).startswith("[") else fa
        q = (r.get("context") or "") + "\n" + r["question"] if r.get("context") else r["question"]
        tasks.append(dict(id=f"olympiad-{r['id']}", source="olympiad", family="math", difficulty_hint=5,
                          prompt=MATH_INSTR + q, answer=str(fa).strip("$ ")))

    for r in take(rng, jl("HumanEval.jsonl"), PLAN["humaneval"]):
        tasks.append(dict(id=r["task_id"].replace("/", "-").lower(), source="humaneval", family="code", difficulty_hint=2,
                          prompt=CODE_INSTR + "Complete this function, keeping its name and signature:\n\n```python\n"
                          + r["prompt"] + "```",
                          code_prompt=r["prompt"], test=r["test"], entry_point=r["entry_point"]))

    mbpp = [r for r in jl("mbpp.jsonl") if 11 <= r["task_id"] <= 510]  # official test split
    for r in take(rng, mbpp, PLAN["mbpp"]):
        tasks.append(dict(id=f"mbpp-{r['task_id']}", source="mbpp", family="code", difficulty_hint=2,
                          prompt=CODE_INSTR + r["text"] + "\nYour code must pass this test:\n" + r["test_list"][0],
                          tests=r["test_list"], test_setup=r.get("test_setup_code", "")))

    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    OUT.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks))
    from collections import Counter
    print(len(tasks), "tasks ->", OUT)
    print(Counter(t["source"] for t in tasks))


if __name__ == "__main__":
    main()
