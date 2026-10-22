"""BeamDial Space: see what each reasoning level costs on Beam, and get a level for your own prompt."""
import json
from pathlib import Path

import gradio as gr

from beamdial import LEVELS, BeamDial, Router

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SUMMARY = json.loads((RESULTS / "summary.json").read_text())
ROUTER = Router.load()

EXAMPLES = [
    "Solve the following problem. Put only the final answer inside \\boxed{}.\n\nA shop sells pens at $3 each. Mia buys 7 pens and pays with a $50 bill. How much change does she get?",
    "Solve the following problem. Put only the final answer inside \\boxed{}.\n\nFind the number of ordered pairs of positive integers (a, b) with a + b = 1000 such that neither a nor b has a zero digit.",
    "Write a correct Python solution. Return the complete code in a single ```python code block.\n\nWrite a function that returns the second largest distinct value in a list, or None if it does not exist.",
]


def policy_table():
    keep = ["fixed: low", "fixed: medium", "fixed: high", "fixed: xhigh", "fixed: max",
            "low, high if low thinks >3000 tok", "majority vote of 3 at low", "agreement cascade",
            "router cv5 + check @ 0.9", "router loso + check @ 0.9", "oracle (upper bound)"]
    rows = []
    for name in keep:
        r = next((p for p in SUMMARY["policies"] if p["policy"] == name), None)
        if r:
            rows.append([name, f"{100 * r['accuracy']:.1f}%", f"{r['tokens_per_task']:,.0f}",
                         f"${r['usd_per_1k']:.2f}", f"{r['calls_per_task']:.2f}"])
    return rows


def recommend(prompt, target):
    if not prompt.strip():
        return "Paste a prompt first.", []
    c = ROUTER.explain(prompt, target)
    pick = c.effort if c.in_domain else "medium"
    rows = [[l, f"{100 * c.probs[l]:.0f}%", f"{c.tokens[l]:,.0f}", "<- pick" if l == pick else ""] for l in LEVELS]
    if not c.in_domain:
        return ("This prompt doesn't look like math or code, which is what the router was trained on, so its "
                "estimates would be guesses. BeamDial uses its fallback level: **`medium`**."), rows
    saved = 1 - c.tokens[pick] / max(c.tokens["max"], 1)
    msg = (f"**Use `reasoning_effort=\"{pick}\"`**. Estimated chance of a correct answer {100 * c.p_correct:.0f}%, "
           f"about {100 * saved:.0f}% fewer completion tokens than `max` on average for this kind of task.")
    return msg, rows


def run_live(prompt, target, token):
    """Optional: runs the prompt on Beam with the visitor's own HF token. The token is used for this call only."""
    if not token or not token.startswith("hf_"):
        return "Add your own Hugging Face token to run this live. It is used for this one call and never stored."
    from openai import OpenAI
    client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=token.strip())
    res = BeamDial(client, target=target).create(messages=[{"role": "user", "content": prompt}],
                                                 max_completion_tokens=16000)
    steps = ", ".join(f"{a.effort} ({a.completion_tokens:,} tok)" for a in res.attempts)
    return f"**Levels run:** {steps}\n\n---\n\n{res.content or '(no answer)'}"


with gr.Blocks(title="BeamDial") as demo:
    gr.Markdown(
        "# BeamDial\nBeam has five reasoning levels. On standard math and code, `low` matches `max` for about half the tokens. "
        f"We ran **{SUMMARY['n_tasks']} graded tasks at every level** and trained a tiny router that picks the "
        "cheapest level likely to get it right. Your prompts may differ: `python -m beamdial calibrate` measures it for you.")
    with gr.Tab("Pick a level"):
        prompt = gr.Textbox(label="Your prompt", lines=7, value=EXAMPLES[1])
        target = gr.Slider(0.5, 0.97, value=0.85, step=0.01, label="Quality target (higher = safer, more tokens)")
        btn = gr.Button("Recommend a level", variant="primary")
        out_md = gr.Markdown()
        out_tab = gr.Dataframe(headers=["level", "est. P(correct)", "mean tokens", ""], interactive=False)
        gr.Examples(EXAMPLES, inputs=prompt)
        btn.click(recommend, [prompt, target], [out_md, out_tab])
        with gr.Accordion("Run it on Beam (uses your own HF token)", open=False):
            token = gr.Textbox(label="HF token", type="password")
            live_btn = gr.Button("Run with BeamDial")
            live_out = gr.Markdown()
            live_btn.click(run_live, [prompt, target, token], live_out)
    with gr.Tab("Results"):
        gr.Image(str(HERE / "figures" / "01-hero.png"), show_label=False)
        gr.Image(str(HERE / "figures" / "03-policies.png"), show_label=False)
        gr.Dataframe(policy_table(), headers=["Policy", "Accuracy", "Tokens / task", "$ / 1k tasks", "Calls / task"],
                     interactive=False)
        gr.Image(str(RESULTS / "levels.png"), show_label=False)
        gr.Markdown((RESULTS / "report.md").read_text())
    with gr.Tab("Use it in code"):
        gr.Markdown(
            "```bash\npip install \"git+https://github.com/CoderRahul01/beamdial\"\n```\n"
            "```python\nfrom openai import OpenAI\nfrom beamdial import BeamDial\n\n"
            "beam = BeamDial(OpenAI(base_url=\"https://router.huggingface.co/v1\", api_key=HF_TOKEN))\n"
            "res = beam.create(messages=[{\"role\": \"user\", \"content\": prompt}],\n"
            "                  verify=lambda text: my_tests_pass(text))  # optional: escalate only on failure\n"
            "print(res.effort, res.total_tokens, res.content)\n```\n"
            "**Measure the gap on your own prompts** (JSONL with `prompt` and `answer`):\n"
            "```bash\npython -m beamdial calibrate my_prompts.jsonl --levels low,high,max --budget-usd 2\n```")

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
