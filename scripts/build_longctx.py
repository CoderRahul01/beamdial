"""Synthetic long-context state-tracking tasks. Contamination-free, graded exactly, difficulty you can dial.

Each task is a long service log with assignment lines like

    2026-03-14T09:12:44Z [billing] set k_amber4 = k_slate1 + 17

hidden among thousands of ordinary log lines. The model must process the log top to bottom like a program and
report the final value of one variable. Variables get reassigned, sometimes before and sometimes after they are
used, so the model has to track state over time instead of grabbing the first match. That is the same failure
production agents hit with stale memory.

Gold answers come from actually executing the program, so they are correct by construction.

python scripts/build_longctx.py   ->  data/tasks_longctx.jsonl
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "tasks_longctx.jsonl"
SEED = 11
CHARS_PER_TOKEN = 4  # rough, for sizing the haystack

# (source, approx tokens, number of assignment ops, minimum dependency depth of the target), 10 tasks each
TIERS = [
    ("longctx_8k", 8_000, 25, 3),
    ("longctx_8k", 8_000, 60, 6),
    ("longctx_8k", 8_000, 120, 10),
    ("longctx_32k", 32_000, 25, 3),
    ("longctx_32k", 32_000, 60, 6),
    ("longctx_32k", 32_000, 120, 10),
]
PER_TIER = 10

SERVICES = ["billing", "auth", "search", "ingest", "notify", "ledger", "gateway", "scheduler", "cache", "export"]
WORDS = ["amber", "slate", "coral", "birch", "cedar", "delta", "ember", "frost", "grove", "haze", "iris", "jade",
         "kiln", "lumen", "mesa", "nova", "onyx", "pike", "quill", "reef", "sable", "tide", "umber", "vale"]
FILLER = [
    "GET /api/v2/orders/{n} 200 {ms}ms",
    "POST /api/v2/checkout 201 {ms}ms request_id={hex}",
    "cache hit ratio={pct}% keys={n}",
    "worker-{w} heartbeat ok queue_depth={d}",
    "retrying upstream call attempt={a} backoff={ms}ms",
    "user session refreshed uid={n} ttl=3600",
    "flushed {n} events to sink in {ms}ms",
    "config reload skipped: no changes (rev {hex})",
    "slow query detected duration={ms}ms table=orders_{w}",
    "metrics pushed series={n} dropped=0",
]

INSTR = (
    "Below is a service log. Lines containing `set <name> = <expression>` are assignments. Process the log from "
    "top to bottom like a program: each assignment uses the values its variables have at that moment, and a later "
    "assignment to the same name replaces the earlier value. Ignore every other line.\n\n"
    "Question: after the whole log is processed, what is the final value of `{target}`?\n"
    "Put only the final integer inside \\boxed{{}}.\n\n<log>\n{log}\n</log>\n\n"
    "Question again: what is the final value of `{target}`? Answer with \\boxed{{}}."
)


def make_program(rng, n_ops, min_depth):
    """Random straight-line program. Returns ops, final values, and the depth of each variable's last value."""
    names = [f"k_{w}{i}" for w in WORDS for i in range(1, 6)]
    rng.shuffle(names)
    pool = names[: max(12, n_ops // 3)]
    val, depth, ops = {}, {}, []
    for name in pool[:6]:  # seed constants
        v = rng.randint(2, 60)
        ops.append((name, str(v)))
        val[name], depth[name] = v, 1
    while len(ops) < n_ops:
        tgt = rng.choice(pool)
        defined = list(val)
        kind = rng.random()
        if kind < 0.55:
            a, c = rng.choice(defined), rng.randint(1, 30)
            sign = rng.choice(["+", "-"])
            expr, v, d = f"{a} {sign} {c}", val[a] + c if sign == "+" else val[a] - c, depth[a] + 1
        elif kind < 0.85:
            a, b = rng.sample(defined, 2) if len(defined) > 1 else (defined[0], defined[0])
            sign = rng.choice(["+", "-"])
            expr, v, d = f"{a} {sign} {b}", val[a] + val[b] if sign == "+" else val[a] - val[b], max(depth[a], depth[b]) + 1
        else:
            v = rng.randint(2, 60)
            expr, d = str(v), 1
        ops.append((tgt, expr))
        val[tgt], depth[tgt] = v, d
    deep = [n for n in val if depth[n] >= min_depth and abs(val[n]) < 10_000]
    if not deep:
        return None
    target = max(deep, key=lambda n: (depth[n], rng.random()))
    # how many times the target was assigned: reassignments are the stale-value trap
    n_assign = sum(1 for t, _ in ops if t == target)
    return ops, target, val[target], depth[target], n_assign


def render_log(rng, ops, approx_tokens):
    budget = approx_tokens * CHARS_PER_TOKEN
    t = 1_773_000_000 + rng.randint(0, 10_000_000)
    lines, n_ops = [], len(ops)
    # spread ops evenly through the haystack
    total_lines = max(n_ops * 3, budget // 70)
    slots = sorted(rng.sample(range(total_lines), n_ops))
    it = iter(ops)
    slot_set = set(slots)
    for i in range(total_lines):
        t += rng.randint(0, 4)
        ts = f"2026-03-{1 + (t // 86400) % 28:02d}T{(t // 3600) % 24:02d}:{(t // 60) % 60:02d}:{t % 60:02d}Z"
        svc = rng.choice(SERVICES)
        if i in slot_set:
            name, expr = next(it)
            lines.append(f"{ts} [{svc}] set {name} = {expr}")
        else:
            f = rng.choice(FILLER).format(n=rng.randint(1, 99999), ms=rng.randint(1, 900), hex=f"{rng.getrandbits(32):08x}",
                                          pct=rng.randint(40, 99), w=rng.randint(1, 16), d=rng.randint(0, 50), a=rng.randint(1, 5))
            lines.append(f"{ts} [{svc}] {f}")
    return "\n".join(lines)


def main():
    rng = random.Random(SEED)
    tasks, k = [], 0
    for source, tokens, n_ops, min_depth in TIERS:
        made = 0
        while made < PER_TIER:
            prog = make_program(rng, n_ops, min_depth)
            if prog is None:
                continue
            ops, target, gold, depth, n_assign = prog
            log = render_log(rng, ops, tokens)
            k += 1
            made += 1
            tasks.append(dict(id=f"{source}-{n_ops}ops-{made:02d}", source=source, family="math", difficulty_hint=3 + (n_ops > 50) + (n_ops > 100),
                              prompt=INSTR.format(target=target, log=log), answer=str(gold),
                              meta=dict(approx_tokens=tokens, n_ops=n_ops, target_depth=depth, target_assignments=n_assign)))
    OUT.write_text("".join(json.dumps(t) + "\n" for t in tasks))
    sizes = [len(t["prompt"]) // CHARS_PER_TOKEN for t in tasks]
    print(f"{len(tasks)} tasks -> {OUT}; approx prompt tokens min={min(sizes):,} max={max(sizes):,} total={sum(sizes):,}")


if __name__ == "__main__":
    main()
