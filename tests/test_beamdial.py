"""Unit tests for the BeamDial wrapper and router. No network, no API key: a fake client stands in for Beam.

    python -m pytest -q
"""
from types import SimpleNamespace as NS

import pytest

from beamdial import LEVELS, BeamDial, Router
from beamdial.client import REASONING_PLACEHOLDER
from beamdial.features import DIM


def tiny_router(pick_from="high"):
    """A router whose estimates jump from 0.1 to 0.95 at `pick_from`, for every prompt."""
    i0 = LEVELS.index(pick_from)
    w = {"levels": LEVELS, "coef": {l: {} for l in LEVELS},
         "intercept": {l: (3.0 if i >= i0 else -2.2) for i, l in enumerate(LEVELS)},
         "tokens": {"all": {l: 1000 * (i + 1) for i, l in enumerate(LEVELS)}}, "default_target": 0.85}
    return Router(w)


class FakeBeam:
    """Answers from a script: a list of (finish_reason, content, tool_calls) per call."""

    def __init__(self, script, reasoning="thinking"):
        self.script, self.calls, self.reasoning = list(script), [], reasoning
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        finish, content, tools = self.script.pop(0)
        msg = NS(content=content, tool_calls=tools, reasoning_content=self.reasoning, model_extra={})
        usage = NS(prompt_tokens=10, completion_tokens=100, completion_tokens_details=NS(reasoning_tokens=80))
        return NS(choices=[NS(message=msg, finish_reason=finish)], usage=usage)


MSG = [{"role": "user", "content": "Solve: how many integers between 1 and 100 are prime?"}]


def test_router_pick_and_monotone():
    r = tiny_router("high")
    c = r.explain("find the integer x with x + 2 = 5")
    assert c.effort == "high"
    vals = [c.probs[l] for l in LEVELS]
    assert vals == sorted(vals)  # higher levels never get lower estimates


def test_family_detection():
    r = tiny_router()
    assert r.family("Write a python function that reverses a list") == "code"
    assert r.family("How many positive divisors does 3600 have?") == "math"
    assert r.family("Write a friendly email to my landlord about the heater") == "other"


def test_out_of_domain_uses_fallback():
    fake = FakeBeam([("stop", "Dear landlord...", None)])
    beam = BeamDial(fake, router=tiny_router("max"), fallback="medium")
    res = beam.create(messages=[{"role": "user", "content": "Write a short thank you note to my team"}])
    assert fake.calls[0]["reasoning_effort"] == "medium"
    assert res.ok


def test_router_level_used_and_no_escalation_on_success():
    fake = FakeBeam([("stop", "\\boxed{25}", None)])
    res = BeamDial(fake, router=tiny_router("high")).create(messages=MSG)
    assert [c["reasoning_effort"] for c in fake.calls] == ["high"]
    assert res.effort == "high" and not res.escalated


def test_verify_failure_escalates_one_level_at_a_time():
    fake = FakeBeam([("stop", "bad", None), ("stop", "bad", None), ("stop", "good", None)])
    res = BeamDial(fake, router=tiny_router("low"), max_attempts=5).create(messages=MSG, verify=lambda t: t == "good")
    assert [c["reasoning_effort"] for c in fake.calls] == ["low", "medium", "high"]
    assert res.escalated and res.ok and res.total_tokens == 300


def test_max_attempts_caps_spend():
    fake = FakeBeam([("stop", "bad", None)] * 5)
    res = BeamDial(fake, router=tiny_router("low"), max_attempts=2).create(messages=MSG, verify=lambda t: False)
    assert len(fake.calls) == 2 and not res.ok


def test_tool_call_reply_is_success_not_failure():
    tool = [NS(id="c1", type="function", function=NS(name="lookup", arguments="{}"))]
    fake = FakeBeam([("tool_calls", None, tool)])
    res = BeamDial(fake, router=tiny_router("low")).create(messages=MSG, verify=lambda t: False)
    assert len(fake.calls) == 1 and res.ok and res.tool_calls


def test_truncation_retries_same_level_with_bigger_budget():
    fake = FakeBeam([("length", None, None), ("stop", "\\boxed{25}", None)])
    res = BeamDial(fake, router=tiny_router("high")).create(messages=MSG, max_completion_tokens=4000)
    assert [c["reasoning_effort"] for c in fake.calls] == ["high", "high"]
    assert [c["max_completion_tokens"] for c in fake.calls] == [4000, 8000]
    assert res.ok


def test_truncation_at_cap_stops():
    fake = FakeBeam([("length", None, None)])
    res = BeamDial(fake, router=tiny_router("high"), max_tokens_cap=4000).create(messages=MSG, max_completion_tokens=4000)
    assert len(fake.calls) == 1 and not res.ok and res.attempts[-1].outcome == "truncated"


def test_to_message_always_has_reasoning():
    fake = FakeBeam([("stop", "hi", None)], reasoning=None)
    res = BeamDial(fake, router=tiny_router("low")).create(messages=MSG)
    m = res.to_message()
    assert m["role"] == "assistant" and m["reasoning_content"] == REASONING_PLACEHOLDER


def test_fixed_policy_never_escalates():
    fake = FakeBeam([("stop", "bad", None)])
    BeamDial(fake, policy="fixed", start="xhigh").create(messages=MSG, verify=lambda t: False)
    assert [c["reasoning_effort"] for c in fake.calls] == ["xhigh"]


def test_reasoning_effort_kwarg_rejected():
    with pytest.raises(ValueError):
        BeamDial(FakeBeam([]), router=tiny_router()).create(messages=MSG, reasoning_effort="max")


def test_cost_estimate():
    fake = FakeBeam([("stop", "ok", None)])
    res = BeamDial(fake, router=tiny_router("low"), prices=(0.4, 1.2)).create(messages=MSG)
    assert res.cost_usd == pytest.approx(10 * 0.4 / 1e6 + 100 * 1.2 / 1e6)


def test_features_deterministic():
    from beamdial.features import featurize
    a, b = featurize("Find all primes p"), featurize("Find all primes p")
    assert a == b and all(k < DIM + 100 for k in a)


class FakeBeamTokens(FakeBeam):
    """Like FakeBeam but each scripted reply also sets reasoning tokens."""

    def _create(self, **kw):
        self.calls.append(kw)
        finish, content, tools, rt = self.script.pop(0)
        msg = NS(content=content, tool_calls=tools, reasoning_content="t", model_extra={})
        usage = NS(prompt_tokens=10, completion_tokens=rt + 20, completion_tokens_details=NS(reasoning_tokens=rt))
        return NS(choices=[NS(message=msg, finish_reason=finish)], usage=usage)


def test_long_think_reasks_at_high():
    fake = FakeBeamTokens([("stop", "\\boxed{1}", None, 5000), ("stop", "\\boxed{2}", None, 7000)])
    res = BeamDial(fake, router=tiny_router("low"), long_think_tokens=3000).create(messages=MSG)
    assert [c["reasoning_effort"] for c in fake.calls] == ["low", "high"]
    assert res.content == "\\boxed{2}" and res.ok and res.attempts[0].outcome == "long_think"


def test_short_think_stays_low():
    fake = FakeBeamTokens([("stop", "\\boxed{1}", None, 400)])
    res = BeamDial(fake, router=tiny_router("low"), long_think_tokens=3000).create(messages=MSG)
    assert len(fake.calls) == 1 and res.effort == "low"


def test_long_think_off_when_verify_given():
    fake = FakeBeamTokens([("stop", "good", None, 9000)])
    BeamDial(fake, router=tiny_router("low"), long_think_tokens=3000).create(messages=MSG, verify=lambda t: True)
    assert len(fake.calls) == 1
