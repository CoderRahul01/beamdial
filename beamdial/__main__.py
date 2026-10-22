"""Command line.

    python -m beamdial "How many primes are below 100?"      # local pick, nothing is sent anywhere
    python -m beamdial --target 0.95 < prompt.txt
    python -m beamdial calibrate my_prompts.jsonl            # measure the gap on YOUR prompts (calls the API)
"""
import argparse
import sys

from .router import LEVELS, Router


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        from .calibrate import main as calibrate_main
        return calibrate_main(sys.argv[2:])
    ap = argparse.ArgumentParser(prog="beamdial", description="Pick Beam's reasoning_effort for a prompt. "
                                 "Use 'python -m beamdial calibrate prompts.jsonl' to measure the gap on your own prompts.")
    ap.add_argument("prompt", nargs="?", help="prompt text (or pipe it on stdin)")
    ap.add_argument("--target", type=float, default=None, help="quality target 0..1 (default from the weights, 0.85)")
    args = ap.parse_args()
    text = args.prompt if args.prompt is not None else sys.stdin.read()
    if not text.strip():
        ap.error("give a prompt as an argument or on stdin")
    c = Router.load().explain(text, args.target)
    print(f"family: {c.family}{'' if c.in_domain else '  (outside training data: BeamDial would use its fallback level)'}")
    print(f"{'level':8}{'P(correct)':>12}{'mean tokens':>16}")
    for lvl in LEVELS:
        mark = "  <- pick" if lvl == c.effort else ""
        print(f"{lvl:8}{100 * c.probs[lvl]:>11.0f}%{c.tokens[lvl]:>16,.0f}{mark}")


if __name__ == "__main__":
    main()
