"""Simulate reasoning-level policies offline from graded runs and write the report.

Because every task was run at every level, any policy (fixed level, router, escalation cascade)
can be replayed without new API calls: we look up what that level actually produced.

python scripts/analyze.py data/graded.jsonl results/
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from beamdial.router import LEVELS  # noqa: E402
from graders import _clean, load_tasks  # noqa: E402
from train_router import GROUPS  # noqa: E402

GROUP_ORDER = ["gsm8k", "math", "competition", "humaneval", "mbpp", "frontier", "longctx"]
GROUP_LABEL = {"gsm8k": "GSM8K (easy math)", "math": "MATH L1-5", "competition": "AMC / AIME 2024 / Olympiad",
               "humaneval": "HumanEval", "mbpp": "MBPP", "frontier": "AIME 2026 / HMMT 2026 / Apex",
               "longctx": "Long-context state tracking"}


def same_answer(a, b):
    if a is None or b is None:
        return False
    a, b = _clean(a), _clean(b)
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        pass
    try:
        from math_verify import parse, verify
        return bool(verify(parse("$" + a + "$"), parse("$" + b + "$")))
    except Exception:
        return False


def load(path):
    tasks = load_tasks(ROOT)
    cell = defaultdict(lambda: defaultdict(list))
    for r in map(json.loads, open(path)):
        cell[r["task_id"]][r["effort"]].append(r)
    ids = sorted(t for t in cell if all(l in cell[t] for l in LEVELS))
    D = {}
    for t in ids:
        D[t] = {}
        for l in LEVELS:
            rs = sorted(cell[t][l], key=lambda r: r.get("sample", 0))
            r0 = rs[0]
            shape_fail = r0["finish_reason"] == "length" or r0["grade_reason"] in ("no_boxed_answer", "no_code_block")
            D[t][l] = dict(
                c=mean(float(r["correct"]) for r in rs),
                k=mean(r["usage"]["completion_tokens"] for r in rs),
                rk=mean((r["usage"].get("reasoning_tokens") or 0) for r in rs),
                pin=mean(r["usage"]["prompt_tokens"] for r in rs),
                lat=mean(r["latency_s"] for r in rs),
                shape_fail=shape_fail,
                pub=r0.get("public_pass"),
                pred=r0.get("pred"),
                trunc=r0["finish_reason"] == "length",
                samples=[(r.get("pred"), float(r["correct"]), r["usage"]["completion_tokens"]) for r in rs],
            )
    return tasks, ids, D


class Sim:
    def __init__(self, tasks, ids, D, pin, pout):
        self.tasks, self.ids, self.D, self.pin, self.pout = tasks, ids, D, pin, pout

    def usd(self, t, levels):
        return sum(self.D[t][l]["pin"] * self.pin / 1e6 + self.D[t][l]["k"] * self.pout / 1e6 for l in levels)

    def summarize(self, name, picks, kind, acc_override=None):
        """picks: {task: (final_level, [levels run])}"""
        acc = mean((acc_override[t] if acc_override else self.D[t][fl]["c"]) for t, (fl, _) in picks.items())
        tok = mean(sum(self.D[t][l]["k"] for l in run) for t, (_, run) in picks.items())
        lat = mean(sum(self.D[t][l]["lat"] for l in run) for t, (_, run) in picks.items())
        usd = mean(self.usd(t, run) for t, (_, run) in picks.items())
        calls = mean(len(run) for _, run in picks.values())
        mix = defaultdict(int)
        for fl, _ in picks.values():
            mix[fl] += 1
        return dict(policy=name, kind=kind, accuracy=acc, tokens_per_task=tok, usd_per_1k=usd * 1000,
                    latency_s=lat, calls_per_task=calls, level_mix={l: mix[l] for l in LEVELS})

    def fixed(self, l):
        return {t: (l, [l]) for t in self.ids}

    def oracle(self):
        out = {}
        for t in self.ids:
            ok = [l for l in LEVELS if self.D[t][l]["c"] >= 0.5]
            fl = min(ok, key=lambda l: self.D[t][l]["k"]) if ok else "low"
            out[t] = (fl, [fl])
        return out

    def check_ok(self, t, l):
        """What a developer can check without ground truth: the answer is well-formed (not truncated, has an
        answer / code block) and, for code, the visible example test passes."""
        d = self.D[t][l]
        if d["shape_fail"]:
            return False
        if self.tasks[t]["family"] == "code" and d["pub"] is False:
            return False
        return True

    def escalate_from(self, t, start, top="max"):
        run = []
        i = LEVELS.index(start)
        while True:
            l = LEVELS[i]
            run.append(l)
            if self.check_ok(t, l) or l == top:
                return l, run
            i += 1

    def cascade(self, start="low"):
        return {t: self.escalate_from(t, start) for t in self.ids}

    def agree_cascade(self):
        """Math: run cheap levels and stop when two consecutive levels agree. Code: escalate on failed visible test."""
        out = {}
        for t in self.ids:
            if self.tasks[t]["family"] == "code":
                out[t] = self.escalate_from(t, "low")
                continue
            run = [LEVELS[0], LEVELS[1]]
            final = None
            for j in range(1, len(LEVELS)):
                a, b = self.D[t][LEVELS[j - 1]], self.D[t][LEVELS[j]]
                if j > 1:
                    run.append(LEVELS[j])
                if not b["shape_fail"] and same_answer(a["pred"], b["pred"]):
                    final = LEVELS[j]
                    break
            out[t] = (final or LEVELS[-1], run)
        return out

    def vote(self, level, k):
        """Self-consistency: k samples at one level, majority answer (math). Needs runs with --samples >= k.
        Returns None if any task lacks k samples."""
        out, acc_override = {}, {}
        for t in self.ids:
            sm = self.D[t][level]["samples"]
            if len(sm) < k:
                return None
            sm = sm[:k]
            if self.tasks[t]["family"] == "math":
                groups = []
                for pred, c, _ in sm:
                    for g in groups:
                        if same_answer(g[0], pred):
                            g[1].append(c)
                            break
                    else:
                        groups.append((pred, [c]))
                best = max(groups, key=lambda g: len(g[1]))
                acc_override[t] = best[1][0]
            else:
                acc_override[t] = sm[0][1]
            out[t] = (level, [level] * k)
        return out, acc_override

    def long_think(self, T, start="low", to="high"):
        """Run `start`; if it thought for more than T reasoning tokens, re-ask at `to`. No checker needed."""
        out = {}
        for t in self.ids:
            d = self.D[t][start]
            thought = d["rk"] or d["k"]
            out[t] = (to, [start, to]) if thought > T else (start, [start])
        return out

    def router(self, oof, split, target, with_check=False):
        out = {}
        for t in self.ids:
            p = oof[t][split]
            best, sm = 0.0, {}
            for l in LEVELS:  # same monotone smoothing as the shipped router
                best = max(best, p[l])
                sm[l] = best
            pick = next((l for l in LEVELS if sm[l] >= target), None)
            if pick is None:
                top = max(sm.values())
                pick = next(l for l in LEVELS if sm[l] >= top - 1e-9)
            out[t] = self.escalate_from(t, pick) if with_check else (pick, [pick])
        return out


def pareto(points):
    pts = sorted(points, key=lambda p: (p["tokens_per_task"], -p["accuracy"]))
    front, best = [], -1
    for p in pts:
        if p["accuracy"] > best + 1e-12:
            front.append(p)
            best = p["accuracy"]
    return front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graded", nargs="?", default=str(ROOT / "data" / "graded.jsonl"))
    ap.add_argument("out", nargs="?", default=str(ROOT / "results"))
    ap.add_argument("--price-in", type=float, default=0.40)
    ap.add_argument("--price-out", type=float, default=1.20)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tasks, ids, D = load(args.graded)
    S = Sim(tasks, ids, D, args.price_in, args.price_out)
    oof_path = out / "oof.json"
    oof = json.loads(oof_path.read_text()) if oof_path.exists() else None

    res = [S.summarize(f"fixed: {l}", S.fixed(l), "fixed") for l in LEVELS]
    res.append(S.summarize("oracle (upper bound)", S.oracle(), "oracle"))
    res.append(S.summarize("cascade from low (check-based)", S.cascade("low"), "cascade"))
    res.append(S.summarize("cascade from medium (check-based)", S.cascade("medium"), "cascade"))
    res.append(S.summarize("agreement cascade", S.agree_cascade(), "cascade"))
    for lvl in ("low", "medium"):
        v = S.vote(lvl, 3)
        if v:
            res.append(S.summarize(f"majority vote of 3 at {lvl}", v[0], "vote", v[1]))
    for T in (2000, 3000, 4000):
        res.append(S.summarize(f"low, high if low thinks >{T} tok", S.long_think(T), "length"))
    curves = {}
    if oof:
        for split in ("cv5", "loso"):
            for chk in (False, True):
                key = f"router {split}" + (" + check" if chk else "")
                curves[key] = []
                for target in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97]:
                    s = S.summarize(f"{key} @ {target}", S.router(oof, split, target, chk), "router")
                    s["target"] = target
                    curves[key].append(s)
                    res.append(s)

    by_group = {}
    for g in GROUP_ORDER:
        sel = [t for t in ids if GROUPS[tasks[t]["source"]] == g]
        if not sel:
            continue
        by_group[g] = dict(
            n=len(sel),
            accuracy={l: mean(D[t][l]["c"] for t in sel) for l in LEVELS},
            median_tokens={l: median(D[t][l]["k"] for t in sel) for l in LEVELS},
            truncated={l: sum(D[t][l]["trunc"] for t in sel) for l in LEVELS},
        )
    min_level = defaultdict(int)
    for t in ids:
        ok = [l for l in LEVELS if D[t][l]["c"] >= 0.5]
        min_level[ok[0] if ok else "none"] += 1

    flag_path = ROOT / "data" / "flagged_tasks.json"
    flagged = set(json.loads(flag_path.read_text())["tasks"]) if flag_path.exists() else set()
    clean = [t for t in ids if t not in flagged]
    clean_acc = {l: mean(D[t][l]["c"] for t in clean) for l in LEVELS} if clean else {}
    fixed_max = next(r for r in res if r["policy"] == "fixed: max")
    for r in res:
        r["accuracy_vs_max"] = r["accuracy"] - fixed_max["accuracy"]
        r["token_share_vs_max"] = r["tokens_per_task"] / fixed_max["tokens_per_task"] if fixed_max["tokens_per_task"] else None

    summary = dict(n_tasks=len(ids), prices=dict(input=args.price_in, output=args.price_out),
                   policies=res, router_curves=curves, by_group=by_group,
                   cheapest_sufficient_level=dict(min_level),
                   flagged_tasks=sorted(flagged & set(ids)), clean_accuracy=clean_acc, n_clean=len(clean),
                   pareto=[p["policy"] for p in pareto([r for r in res if r["kind"] != "oracle"])])
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    write_report(summary, out)
    plot(summary, out)
    print((out / "report.md").read_text()[:3500])


def fmt_pct(x):
    return f"{100 * x:.1f}%"


def write_report(s, out):
    L = [f"# BeamDial results ({s['n_tasks']} tasks x 5 reasoning levels)\n",
         f"Prices used: ${s['prices']['input']}/M input, ${s['prices']['output']}/M output tokens.\n",
         "## Policies\n", "| Policy | Accuracy | Tokens / task | $ / 1k tasks | Calls / task | vs fixed max |",
         "|---|---|---|---|---|---|"]
    key = [r for r in s["policies"] if r["kind"] != "router" or r.get("target") in (0.8, 0.9)]
    for r in key:
        L.append(f"| {r['policy']} | {fmt_pct(r['accuracy'])} | {r['tokens_per_task']:,.0f} | ${r['usd_per_1k']:.2f} | "
                 f"{r['calls_per_task']:.2f} | {100 * r['accuracy_vs_max']:+.1f} pts, {100 * (r['token_share_vs_max'] or 0):.0f}% of tokens |")
    L += ["\n## Accuracy by level and task group\n", "| Group | n | " + " | ".join(LEVELS) + " |",
          "|---|---|" + "---|" * len(LEVELS)]
    for g, d in s["by_group"].items():
        L.append(f"| {GROUP_LABEL[g]} | {d['n']} | " + " | ".join(fmt_pct(d["accuracy"][l]) for l in LEVELS) + " |")
    if s.get("clean_accuracy"):
        L += [f"\nExcluding {len(s['flagged_tasks'])} tasks with broken reference answers or tests "
              f"(see data/flagged_tasks.json), {s['n_clean']} tasks: "
              + ", ".join(f"{l} {fmt_pct(s['clean_accuracy'][l])}" for l in LEVELS)]
    L += ["\n## Median completion tokens by level\n", "| Group | " + " | ".join(LEVELS) + " |", "|---|" + "---|" * len(LEVELS)]
    for g, d in s["by_group"].items():
        L.append(f"| {GROUP_LABEL[g]} | " + " | ".join(f"{d['median_tokens'][l]:,.0f}" for l in LEVELS) + " |")
    L += ["\n## Cheapest level that solved each task\n",
          ", ".join(f"{k}: {v}" for k, v in s["cheapest_sufficient_level"].items())]
    (out / "report.md").write_text("\n".join(L) + "\n")


def plot(s, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f2328", "#6e7781", "#e6e8eb"
    blue, orange, green, purple = "#2f6fdb", "#e07b39", "#2e9e6a", "#8a63d2"
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False})

    # 1. Accuracy vs tokens
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=150)
    fixed = [r for r in s["policies"] if r["kind"] == "fixed"]
    n = s["n_tasks"]

    def ci95(p):  # Wilson interval half-widths, in points, for n tasks
        z = 1.96
        c = (p + z * z / (2 * n)) / (1 + z * z / n)
        h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
        return 100 * (p - (c - h)), 100 * ((c + h) - p)

    errs = list(zip(*[ci95(r["accuracy"]) for r in fixed]))
    ax.errorbar([r["tokens_per_task"] for r in fixed], [100 * r["accuracy"] for r in fixed], yerr=errs, fmt="-o",
                color=muted, lw=1.5, capsize=3, label="Fixed reasoning level (95% CI)", zorder=2)
    for r in fixed:
        ax.annotate(r["policy"].split(": ")[1], (r["tokens_per_task"], 100 * r["accuracy"]), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=9, color=muted)
    for key, col in (("router cv5 + check", blue), ("router loso + check", purple)):
        c = sorted(s["router_curves"].get(key) or [], key=lambda r: r["tokens_per_task"])
        if c:
            ax.plot([r["tokens_per_task"] for r in c], [100 * r["accuracy"] for r in c], "-", color=col, lw=2,
                    label={"router cv5 + check": "BeamDial router + escalation",
                           "router loso + check": "Router on unseen task types"}[key], zorder=3)
    for r in s["policies"]:
        if r["policy"] == "agreement cascade":
            ax.scatter(r["tokens_per_task"], 100 * r["accuracy"], s=70, color=orange, zorder=4, label="Agreement cascade")
        if r["policy"] == "low, high if low thinks >3000 tok":
            ax.scatter(r["tokens_per_task"], 100 * r["accuracy"], s=80, color="#c2475d", marker="D", zorder=5,
                       label="Low, re-ask at high if it thinks long (no checker)")
        if r["policy"] == "cascade from low (check-based)":
            ax.scatter(r["tokens_per_task"], 100 * r["accuracy"], s=70, color=green, marker="s", zorder=4,
                       label="Escalate from low on failed check")
        if r["kind"] == "oracle":
            ax.scatter(r["tokens_per_task"], 100 * r["accuracy"], s=110, color=ink, marker="*", zorder=4,
                       label="Oracle (cheapest level that works)")
    ax.set_xscale("log")
    lo = min(100 * r["accuracy"] for r in s["policies"])
    ax.set_ylim(max(0, lo - 6), 100.5)
    ax.set_xlabel("Completion tokens per task (log scale)")
    ax.set_ylabel("Tasks solved (%)")
    ax.grid(True, color=grid, lw=0.8)
    ax.set_title("Accuracy vs reasoning tokens on Beam", loc="left", fontsize=13, color=ink, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.text(0.01, 0.02, f"{n} tasks. Differences smaller than the error bars are noise.", transform=ax.transAxes,
            fontsize=8.5, color=muted)
    fig.tight_layout()
    fig.savefig(out / "pareto.png")
    plt.close(fig)

    # 2. Tokens and accuracy by level per group
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    cols = [blue, purple, orange, green, "#c2475d"]
    for (g, d), col in zip(s["by_group"].items(), cols):
        axes[0].plot(LEVELS, [d["median_tokens"][l] for l in LEVELS], "-o", color=col, label=GROUP_LABEL[g], lw=1.8)
        axes[1].plot(LEVELS, [100 * d["accuracy"][l] for l in LEVELS], "-o", color=col, label=GROUP_LABEL[g], lw=1.8)
    axes[0].set_yscale("log")
    axes[0].set_title("Median completion tokens", loc="left", fontweight="bold", color=ink)
    axes[1].set_title("Tasks solved (%)", loc="left", fontweight="bold", color=ink)
    for a in axes:
        a.grid(True, color=grid, lw=0.8)
        a.set_xlabel("reasoning_effort")
    axes[1].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "levels.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
