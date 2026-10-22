"""Command line.

    beamdial "How many primes are below 100?"      # local pick, nothing is sent anywhere
    beamdial --target 0.95 < prompt.txt
    beamdial calibrate my_prompts.jsonl            # measure the gap on your own prompts (calls the API)

`python -m beamdial ...` works the same way.
"""
import argparse
import sys

from .router import LEVELS, Router


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("-V", "--version"):
        from . import __version__
        print(f"beamdial {__version__}")
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        from .calibrate import main as calibrate_main
        return calibrate_main(sys.argv[2:])
    ap = argparse.ArgumentParser(prog="beamdial", description="Pick Beam's reasoning_effort for a prompt. "
                                 "Use 'beamdial calibrate prompts.jsonl' to measure the gap on your own prompts.")
    ap.add_argument("prompt", nargs="?", help="prompt text (or pipe it on stdin)")
    ap.add_argument("--target", type=float, default=None, help="quality target 0..1 (default from the weights, 0.85)")
    args = ap.parse_args()
    text = args.prompt if args.prompt is not None else sys.stdin.read()
    if not text.strip():
        ap.error("give a prompt as an argument or on stdin")
    c = Router.load().explain(text, args.target)
    pick = c.effort if c.in_domain else "medium"
    if c.in_domain:
        print(f"family: {c.family}")
    else:
        print("family: other. The router was trained on math and code, so BeamDial uses its fallback level (medium).")
    print(f"{'level':8}{'P(correct)':>12}{'mean tokens':>16}")
    for lvl in LEVELS:
        mark = "  <- pick" if lvl == pick else ""
        print(f"{lvl:8}{100 * c.probs[lvl]:>11.0f}%{c.tokens[lvl]:>16,.0f}{mark}")


if __name__ == "__main__":
    main()
