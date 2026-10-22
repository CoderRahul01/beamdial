"""Check that the browser router (space_static/router.js) gives the same answers as the Python router."""
import json, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from beamdial.router import Router, LEVELS  # noqa: E402

prompts = [t["prompt"] for t in map(json.loads, open(ROOT / "data" / "tasks.jsonl"))]
prompts += ["Write a short thank you note to my team", "héllo wörld 😀 12345 $x$ \\frac{1}{2} ??",
            "", "def f(x):\n    return x\n```python```", "Find all integers n such that n^2 + 1 divides n^3 + 7"]
r = Router.load()
py = [dict(effort=r.explain(p).effort, probs=r.probs(p), family=r.family(p)) for p in prompts]
js_code = f"""
import {{ Router }} from '{(ROOT / 'space_static' / 'router.js').as_posix()}';
import fs from 'fs';
const w = JSON.parse(fs.readFileSync('{(ROOT / 'beamdial' / 'router_weights.json').as_posix()}', 'utf8'));
const prompts = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const r = new Router(w);
console.log(JSON.stringify(prompts.map(p => {{ const e = r.explain(p); return {{effort: e.effort, probs: r.probs(p), family: e.family}}; }})));
"""
with tempfile.TemporaryDirectory() as d:
    (Path(d) / "p.json").write_text(json.dumps(prompts))
    (Path(d) / "t.mjs").write_text(js_code)
    js = json.loads(subprocess.check_output(["node", str(Path(d) / "t.mjs"), str(Path(d) / "p.json")]))
worst, mism, wi = 0.0, 0, -1
for a, b in zip(py, js):
    mism += (a["effort"] != b["effort"]) + (a["family"] != b["family"])
    d_ = max(abs(a["probs"][l] - b["probs"][l]) for l in LEVELS)
    if d_ > worst: worst, wi = d_, prompts.index(prompts[py.index(a)])
print(f"{len(prompts)} prompts: pick/family mismatches={mism}, max probability difference={worst:.2e}")
if worst > 1e-9: print("worst prompt:", repr(prompts[wi][:300]))
sys.exit(1 if mism or worst > 1e-9 else 0)
