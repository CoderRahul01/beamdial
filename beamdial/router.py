"""BeamDial router: predicts, for a prompt, how likely Beam is to solve it at each reasoning level,
then picks the cheapest level that clears your quality target.

    from beamdial import Router
    r = Router.load()
    r.route("Prove that there are infinitely many primes.")       # -> "high"
    r.explain("...")                                               # per-level probability and token estimate
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .features import featurize

LEVELS = ["low", "medium", "high", "xhigh", "max"]
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "router_weights.json"


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class Choice:
    effort: str
    p_correct: float
    est_tokens: float
    probs: dict
    tokens: dict
    family: str = "math"
    in_domain: bool = True


class Router:
    def __init__(self, weights: dict):
        self.w = weights
        self.levels = weights["levels"]
        self.coef = {lvl: {int(k): v for k, v in weights["coef"][lvl].items()} for lvl in self.levels}
        self.intercept = weights["intercept"]
        # mean completion tokens per level and family, measured on the BeamDial dataset
        self.token_table = weights.get("tokens", {})

    @classmethod
    def load(cls, path=None):
        p = Path(path or DEFAULT_WEIGHTS)
        if not p.exists():
            raise FileNotFoundError(
                f"Router weights not found at {p}. Train them with scripts/train_router.py, "
                "or use BeamDial(client, policy='cascade') which needs no weights.")
        return cls(json.loads(p.read_text()))

    def family(self, prompt: str) -> str:
        """code | math | other. The router was trained on math and Python only; "other" means its
        estimates are guesses, so BeamDial uses a fixed fallback level instead."""
        p = prompt.lower()
        code_hits = sum(k in p for k in ("```", "def ", "python", "function", "return ", "implement", "class "))
        if code_hits >= 1 and ("python" in p or "def " in p or "```" in p or code_hits >= 2):
            return "code"
        math_hits = sum(k in p for k in ("\\", "$", "\\boxed", "solve", "compute", "how many", "find the", "integer",
                                         "probability", "equation", "prove", "sum of", "remainder", "triangle",
                                         "polynomial", "evaluate", "simplify"))
        digits = sum(c.isdigit() for c in p)
        if math_hits >= 1 or digits >= 4:
            return "math"
        return "other"

    def probs(self, prompt: str) -> dict:
        x = featurize(prompt)
        out = {}
        for lvl in self.levels:
            c = self.coef[lvl]
            z = self.intercept[lvl] + sum(v * c.get(i, 0.0) for i, v in x.items())
            out[lvl] = _sigmoid(z)
        # Reasoning more rarely hurts on average, so smooth with a running max: a higher level never
        # gets a lower estimate than a cheaper one. This keeps choices stable for small score noise.
        best = 0.0
        for lvl in self.levels:
            best = max(best, out[lvl])
            out[lvl] = best
        return out

    def tokens(self, prompt: str) -> dict:
        fam = self.family(prompt)
        tab = self.token_table.get(fam) or self.token_table.get("all") or {}
        return {lvl: float(tab.get(lvl, 0)) for lvl in self.levels}

    def explain(self, prompt: str, target: float = None) -> Choice:
        target = self.w.get("default_target", 0.85) if target is None else target
        pr, tk = self.probs(prompt), self.tokens(prompt)
        pick = None
        for lvl in self.levels:
            if pr[lvl] >= target:
                pick = lvl
                break
        if pick is None:  # nothing clears the bar: take the level with the best estimate, cheapest on ties
            top = max(pr.values())
            pick = next(l for l in self.levels if pr[l] >= top - 1e-9)
        fam = self.family(prompt)
        return Choice(pick, pr[pick], tk[pick], pr, tk, fam, fam != "other")

    def route(self, prompt: str, target: float = None) -> str:
        return self.explain(prompt, target).effort
