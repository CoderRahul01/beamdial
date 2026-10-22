"""Render the launch figures (blog/social quality) from results/launch_figures.json.

Each figure is an HTML page with inline SVG, screenshotted at 2x -> 3200x1800 PNG (16:9).
python scripts/make_figures.py   ->  assets/figures/*.png
"""
import asyncio
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "figures"
F = json.loads((ROOT / "results" / "launch_figures.json").read_text())
L = F["levels"]
W, H = 1600, 900

INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#c9c8c1"
FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"

CSS = f"""
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ width:{W}px; height:{H}px; background:{SURF}; font-family:{FONT}; color:{INK};
       -webkit-font-smoothing:antialiased; }}
.page {{ position:relative; width:{W}px; height:{H}px; padding:72px 88px; }}
.eyebrow {{ font-size:20px; font-weight:600; color:{BLUE}; letter-spacing:0.02em; }}
h1 {{ font-size:58px; line-height:1.08; font-weight:700; letter-spacing:-0.025em; margin-top:18px; }}
.sub {{ font-size:23px; line-height:1.45; color:{INK2}; margin-top:20px; }}
.foot {{ position:absolute; left:88px; right:88px; bottom:44px; display:flex; justify-content:space-between;
         font-size:16px; color:{MUTED}; border-top:1px solid {GRID}; padding-top:18px; }}
.foot b {{ color:{INK}; font-weight:700; }}
svg text {{ font-family:{FONT}; }}
"""

FOOT = (f"<div class='foot'><span><b>BeamDial</b> &nbsp;·&nbsp; {F['n_tasks']} tasks × 5 levels = "
        f"{F['n_runs']:,} graded runs on Beam 1.0 (501B A23B) via Hugging Face, Baseten</span>"
        "<span>Error bars: 95% Wilson interval</span></div>")


def fmt(n):
    return f"{n:,.0f}"


def bars(x0, y0, w, h, values, labels, ymax, highlight, title, unit_fmt, ci=None, ytick_step=None, ymin=0):
    """Vertical bar chart in an SVG group. One series; the highlighted bar is blue, the rest gray."""
    n = len(values)
    gap = w / n
    bw = min(64, gap * 0.56)
    sy = lambda v: y0 + h - (v - ymin) / (ymax - ymin) * h  # noqa: E731
    s = [f"<text x='{x0}' y='{y0 - 50}' font-size='19' font-weight='600' fill='{INK}'>{title}</text>"]
    if ytick_step:
        v = ymin
        while v <= ymax + 1e-9:
            y = sy(v)
            s.append(f"<line x1='{x0}' x2='{x0 + w}' y1='{y:.1f}' y2='{y:.1f}' stroke='{GRID}' stroke-width='1'/>")
            s.append(f"<text x='{x0 - 12}' y='{y + 5:.1f}' font-size='14' fill='{MUTED}' text-anchor='end'>{fmt(v)}</text>")
            v += ytick_step
    s.append(f"<line x1='{x0}' x2='{x0 + w}' y1='{y0 + h}' y2='{y0 + h}' stroke='{BASE}' stroke-width='1.5'/>")
    for i, (v, lab) in enumerate(zip(values, labels)):
        cx = x0 + gap * (i + 0.5)
        col = BLUE if lab == highlight else GRAY
        top = sy(v)
        r = 4
        hgt = y0 + h - top
        # rounded data-end, square at the baseline
        s.append(f"<path d='M{cx - bw / 2:.1f},{y0 + h} V{top + r:.1f} Q{cx - bw / 2:.1f},{top:.1f} {cx - bw / 2 + r:.1f},{top:.1f} "
                 f"H{cx + bw / 2 - r:.1f} Q{cx + bw / 2:.1f},{top:.1f} {cx + bw / 2:.1f},{top + r:.1f} V{y0 + h} Z' fill='{col}'/>"
                 if hgt > r else f"<rect x='{cx - bw / 2:.1f}' y='{top:.1f}' width='{bw:.1f}' height='{hgt:.1f}' fill='{col}'/>")
        label_y = top - 12
        if ci:
            lo, hi = ci[i]
            s.append(f"<line x1='{cx}' x2='{cx}' y1='{sy(lo):.1f}' y2='{sy(min(hi, ymax)):.1f}' stroke='{INK2}' stroke-width='1.5'/>")
            for yy in (sy(lo), sy(min(hi, ymax))):
                s.append(f"<line x1='{cx - 7}' x2='{cx + 7}' y1='{yy:.1f}' y2='{yy:.1f}' stroke='{INK2}' stroke-width='1.5'/>")
            label_y = sy(min(hi, ymax)) - 12
        weight = 700 if lab == highlight else 500
        s.append(f"<text x='{cx}' y='{label_y:.1f}' font-size='18' font-weight='{weight}' fill='{INK}' text-anchor='middle'>{unit_fmt(v)}</text>")
        s.append(f"<text x='{cx}' y='{y0 + h + 30}' font-size='16' fill='{INK2 if lab != highlight else INK}' "
                 f"font-weight='{600 if lab == highlight else 400}' text-anchor='middle'>{lab}</text>")
    return "\n".join(s)


def page(body):
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body><div class='page'>{body}{FOOT}</div></body></html>"


FOOT_NOTE = "Error bars: 95% Wilson interval"


def fig_hero():
    o = F["overall"]
    acc = [o[l]["acc"] for l in L]
    tok = [o[l]["tokens"] for l in L]
    ci = [o[l]["ci"] for l in L]
    share = 100 * o["low"]["tokens"] / o["max"]["tokens"]
    speed = o["max"]["latency"] / o["low"]["latency"]
    left = (f"<div style='position:absolute;left:88px;top:150px;width:560px'>"
            f"<div class='eyebrow'>BEAMDIAL · BENCHMARK RESULTS</div>"
            f"<h1 style='font-size:52px'>On standard math and code, Beam's low matches max.</h1>"
            f"<div class='sub'>{F['n_tasks']} graded math and code tasks, each run at all five reasoning levels. "
            f"Every level lands within the same error bars. <b style='color:{INK}'>low</b> gets there with "
            f"{share:.0f}% of the tokens and answers {speed:.1f}× faster. Frontier and long-context tasks are next (v0.2).</div>"
            f"<div style='display:flex;gap:48px;margin-top:44px'>"
            f"<div><div style='font-size:64px;font-weight:700;letter-spacing:-0.03em'>{o['low']['acc']:.1f}%</div>"
            f"<div style='font-size:17px;color:{INK2};margin-top:4px'>accuracy at low<br>(max: {o['max']['acc']:.1f}%)</div></div>"
            f"<div><div style='font-size:64px;font-weight:700;letter-spacing:-0.03em;color:{BLUE}'>−{100 - share:.0f}%</div>"
            f"<div style='font-size:17px;color:{INK2};margin-top:4px'>tokens vs max<br>({fmt(o['low']['tokens'])} vs {fmt(o['max']['tokens'])} per task)</div></div>"
            f"</div></div>")
    svg = (f"<svg style='position:absolute;left:740px;top:96px' width='780' height='680'>"
           + bars(70, 90, 680, 200, acc, L, 100, "low", "Accuracy (%)", lambda v: f"{v:.1f}", ci=ci, ytick_step=25)
           + bars(70, 440, 680, 200, tok, L, 2000, "low", "Completion tokens per task", fmt, ytick_step=500)
           + "</svg>")
    return page(left + svg)


def fig_dial():
    c = F["groups"]["Competition"]
    n = c["n"]
    rt = [c[l]["reasoning"] for l in L]
    acc = [c[l]["acc"] for l in L]
    ci = [c[l]["ci"] for l in L]
    head = (f"<div class='eyebrow'>COMPETITION MATH · AMC, AIME 2024, OLYMPIADBENCH · {n} TASKS</div>"
            f"<h1>The dial works. The accuracy doesn't move.</h1>"
            f"<div class='sub' style='max-width:1300px'>At max, Beam thinks {rt[-1] / rt[0]:.1f}× longer than at low on the hardest problems in the set. "
            f"Accuracy stays inside the same error bars, so the extra thinking buys no measurable gain here.</div>")
    svg = (f"<svg style='position:absolute;left:88px;top:340px' width='1440' height='440'>"
           + bars(70, 80, 560, 270, rt, L, 4000, "max", "Mean reasoning tokens per task", fmt, ytick_step=1000)
           + bars(830, 80, 560, 270, acc, L, 100, "max", "Accuracy (%)", lambda v: f"{v:.1f}", ci=ci, ytick_step=25)
           + "</svg>")
    return page(head + svg)


def fig_frontier():
    p = F["policies"]
    o = F["overall"]
    x0, y0, w, h = 120, 60, 1240, 380
    tmin, tmax, amin, amax = 700, 2500, 92, 100
    import math
    sx = lambda t: x0 + (math.log(t) - math.log(tmin)) / (math.log(tmax) - math.log(tmin)) * w  # noqa: E731
    sy = lambda a: y0 + h - (a - amin) / (amax - amin) * h  # noqa: E731
    s = []
    for a in range(amin, amax + 1, 2):
        s.append(f"<line x1='{x0}' x2='{x0 + w}' y1='{sy(a):.1f}' y2='{sy(a):.1f}' stroke='{GRID}'/>")
        s.append(f"<text x='{x0 - 14}' y='{sy(a) + 5:.1f}' font-size='14' fill='{MUTED}' text-anchor='end'>{a}%</text>")
    for t in (800, 1000, 1500, 2000):
        s.append(f"<line x1='{sx(t):.1f}' x2='{sx(t):.1f}' y1='{y0}' y2='{y0 + h}' stroke='{GRID}'/>")
        s.append(f"<text x='{sx(t):.1f}' y='{y0 + h + 28}' font-size='14' fill='{MUTED}' text-anchor='middle'>{fmt(t)}</text>")
    s.append(f"<line x1='{x0}' x2='{x0 + w}' y1='{y0 + h}' y2='{y0 + h}' stroke='{BASE}' stroke-width='1.5'/>")
    s.append(f"<text x='{x0 + w / 2}' y='{y0 + h + 62}' font-size='16' fill='{INK2}' text-anchor='middle'>Completion tokens per task (log scale) →</text>")
    s.append(f"<text x='{x0 - 70}' y='{y0 - 22}' font-size='16' fill='{INK2}'>Tasks solved</text>")
    pts = [(p[f"fixed: {l}"]["tokens"], p[f"fixed: {l}"]["acc"], l) for l in L]
    s.append("<polyline fill='none' stroke='#9b9a94' stroke-width='2' points='" + " ".join(f"{sx(t):.1f},{sy(a):.1f}" for t, a, _ in pts) + "'/>")
    place = {"low": (0, 32, "middle"), "medium": (0, 32, "middle"), "high": (0, 34, "middle"),
             "xhigh": (0, -18, "middle"), "max": (0, 32, "middle")}
    for t, a, l in pts:
        s.append(f"<circle cx='{sx(t):.1f}' cy='{sy(a):.1f}' r='7' fill='#6f6e69' stroke='{SURF}' stroke-width='2'/>")
        dx, dy, anc = place[l]
        s.append(f"<text x='{sx(t) + dx:.1f}' y='{sy(a) + dy:.1f}' font-size='16' fill='{INK2}' text-anchor='{anc}'>{l} {a:.1f}%</text>")

    def mark(key, color, label, shape="circle", dx=14, dy=-14, anchor="start"):
        t, a = p[key]["tokens"], p[key]["acc"]
        X, Y = sx(t), sy(a)
        if shape == "diamond":
            s.append(f"<path d='M{X},{Y - 11} L{X + 11},{Y} L{X},{Y + 11} L{X - 11},{Y} Z' fill='{color}' stroke='{SURF}' stroke-width='2'/>")
        elif shape == "star":
            pts_ = []
            for k in range(10):
                rr = 13 if k % 2 == 0 else 5.5
                ang = math.pi / 2 + k * math.pi / 5
                pts_.append(f"{X + rr * math.cos(ang):.1f},{Y - rr * math.sin(ang):.1f}")
            s.append(f"<polygon points='{' '.join(pts_)}' fill='{color}'/>")
        else:
            s.append(f"<circle cx='{X:.1f}' cy='{Y:.1f}' r='8' fill='{color}' stroke='{SURF}' stroke-width='2'/>")
        s.append(f"<text x='{X + dx:.1f}' y='{Y + dy:.1f}' font-size='16' font-weight='600' fill='{INK}' text-anchor='{anchor}'>{label}</text>")
        s.append(f"<text x='{X + dx:.1f}' y='{Y + dy + 20:.1f}' font-size='14' fill='{INK2}' text-anchor='{anchor}'>{a:.1f}% · {fmt(t)} tokens</text>")

    mark("low, high if low thinks >3000 tok", ORANGE, "Low, re-ask at high if it thinks long", "diamond", dx=-18, dy=-62, anchor="end")
    mark("cascade from low (check-based)", BLUE, "Low, escalate on failed check", dx=16, dy=-44)
    mark("agreement cascade", "#6f6e69", "Two levels must agree", dx=-16, dy=-36, anchor="end")
    mark("oracle (upper bound)", INK, "Oracle (needs the answer key)", "star", dx=18, dy=-10)
    s.append(f"<rect x='{x0}' y='{sy(98.5):.1f}' width='{w}' height='{sy(94.1) - sy(98.5):.1f}' fill='{BLUE}' opacity='0.05'/>")
    s.append(f"<text x='{x0 + w - 8}' y='{sy(94.1) - 10:.1f}' font-size='14' fill='{MUTED}' text-anchor='end'>"
             "Shaded: the 95% error band of the fixed levels. Everything inside it is a statistical tie.</text>")
    head = ("<div class='eyebrow'>ALL 410 TASKS · EVERY POLICY REPLAYED ON REAL OUTPUTS</div>"
            "<h1 style='font-size:50px'>Same accuracy band. Very different bills.</h1>"
            "<div class='sub' style='max-width:1350px'>Every strategy lands within about two points, inside the error band. "
            "What changes is cost: start at low and escalate only on a signal, and you pay close to low's price, not max's.</div>")
    svg = f"<svg style='position:absolute;left:88px;top:300px' width='1440' height='520'>{''.join(s)}</svg>"
    return page(head + svg)


def fig_signal():
    lt = F["long_think"]
    acc = [lt[l] for l in L]
    p = F["policies"]["low, high if low thinks >3000 tok"]
    mx = F["policies"]["fixed: max"]
    head = (f"<div class='eyebrow'>A FREE DIFFICULTY SIGNAL · NO CHECKER NEEDED</div>"
            f"<h1>When low thinks long, ask again at high.</h1>"
            f"<div class='sub' style='max-width:620px'>On the {lt['n']} tasks where low spent more than "
            f"{fmt(lt['threshold'])} reasoning tokens, it was right {acc[0]:.0f}% of the time. high got all of them.</div>"
            f"<div style='margin-top:40px;max-width:600px;padding:24px 28px;border:1px solid {GRID};border-radius:14px;background:#ffffff'>"
            f"<div style='font-size:17px;color:{INK2}'>As a policy on all {F['n_tasks']} tasks</div>"
            f"<div style='font-size:40px;font-weight:700;margin-top:6px;letter-spacing:-0.02em'>{p['acc']:.1f}% "
            f"<span style='font-size:22px;font-weight:500;color:{INK2}'>at {100 * p['tokens'] / mx['tokens']:.0f}% of max's tokens</span></div>"
            f"<div style='font-size:16px;color:{MUTED};margin-top:8px'>BeamDial: <code style='font-size:15px'>BeamDial(client, long_think_tokens=3000)</code></div>"
            f"</div>")
    head = f"<div style='position:absolute;left:88px;top:120px;width:640px'>{head}</div>"
    svg = (f"<svg style='position:absolute;left:800px;top:190px' width='720' height='520'>"
           + bars(60, 70, 620, 380, acc, L, 100, "high", f"Accuracy on the {lt['n']} long-thinking tasks (%)",
                  lambda v: f"{v:.0f}", ytick_step=25)
           + f"<text x='60' y='510' font-size='14' fill='{MUTED}'>Small sample (n = {lt['n']}). v0.2 retests this on frontier problems.</text>"
           + "</svg>")
    return page(head + svg)


FIGS = {"01-hero": fig_hero, "02-dial-works": fig_dial, "03-policies": fig_frontier, "04-long-think-signal": fig_signal}


async def main():
    from playwright.async_api import async_playwright
    OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        b = await pw.chromium.launch()
        pg = await b.new_page(viewport={"width": W, "height": H}, device_scale_factor=2)
        for name, fn in FIGS.items():
            html = fn()
            (OUT / f"{name}.html").write_text(html)
            await pg.set_content(html)
            await pg.wait_for_timeout(150)
            await pg.screenshot(path=str(OUT / f"{name}.png"))
            print("wrote", OUT / f"{name}.png")
        await b.close()


if __name__ == "__main__":
    asyncio.run(main())
