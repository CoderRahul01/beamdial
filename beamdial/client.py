"""Drop-in wrapper around any OpenAI-compatible client that serves Beam.

    from openai import OpenAI
    from beamdial import BeamDial

    beam = BeamDial(OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN))
    res = beam.create(messages=[{"role": "user", "content": "..."}])
    print(res.content, res.effort, res.total_tokens)

    # Escalate only when a check you already have fails (tests, JSON schema, regex):
    res = beam.create(messages=msgs, verify=lambda text: run_my_tests(text))

    # Continue the conversation: Beam needs reasoning_content on every assistant turn you send back.
    msgs.append(res.to_message())

Edge cases handled here (see docs/03-edge-cases-and-tradeoffs.md):
- Tool-call replies have no text content. They count as a success, never as a failure to escalate.
- A reply cut off by max_completion_tokens is retried at the SAME level with a bigger budget.
  Climbing to a higher level would think even longer and get cut off again.
- Beam rejects assistant turns without reasoning_content. to_message() always includes it, with a
  placeholder when the provider returned none (the same fix Reflection documents for Hermes).
- Prompts outside what the router was trained on (not math, not code) get a fixed fallback level.
- max_attempts caps the extra calls, so a stubborn failure cannot run up the bill.
- long_think_tokens (opt-in): if Beam thinks unusually long at a cheap level, that is a sign the prompt is hard.
  BeamDial re-asks once at `long_think_to` (default "high"). Needs no checker. On the v0.1 data, "low, and high
  if low used over 3,000 tokens" matched or beat every fixed level.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

from .router import LEVELS, Router

DEFAULT_MODEL = "reflection/Beam-1.0-501B-A23B"
REASONING_PLACEHOLDER = "(reasoning not returned by provider)"


@dataclass
class Attempt:
    effort: str
    content: Optional[str]
    finish_reason: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: Optional[int]
    max_completion_tokens: Optional[int]
    passed: Optional[bool]
    outcome: str  # ok | tool_call | verify_failed | truncated | empty | long_think


@dataclass
class DialResult:
    response: object
    content: Optional[str]
    effort: str
    routed_effort: str
    attempts: list = field(default_factory=list)
    prices: Optional[tuple] = None

    @property
    def message(self):
        return self.response.choices[0].message

    @property
    def tool_calls(self):
        return getattr(self.message, "tool_calls", None)

    @property
    def reasoning(self) -> Optional[str]:
        m = self.message
        r = getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None)
        extra = getattr(m, "model_extra", None) or {}
        return r or extra.get("reasoning_content") or extra.get("reasoning")

    @property
    def total_tokens(self) -> int:
        """Completion tokens summed over every call, including failed attempts. This is what you pay for."""
        return sum(a.completion_tokens for a in self.attempts)

    @property
    def escalated(self) -> bool:
        return self.effort != self.routed_effort

    @property
    def ok(self) -> bool:
        return self.attempts[-1].outcome in ("ok", "tool_call", "long_think")

    @property
    def cost_usd(self) -> Optional[float]:
        if not self.prices:
            return None
        p_in, p_out = self.prices
        return sum(a.prompt_tokens * p_in / 1e6 + a.completion_tokens * p_out / 1e6 for a in self.attempts)

    def to_message(self) -> dict:
        """The assistant turn to append to your history. Always carries a non-empty reasoning_content,
        which Beam requires on every assistant message you send back."""
        msg = {"role": "assistant", "content": self.content or "",
               "reasoning_content": self.reasoning or REASONING_PLACEHOLDER}
        if self.tool_calls:
            msg["tool_calls"] = [tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in self.tool_calls]
        return msg


class BeamDial:
    """policy:
    - "router":  start at the level the router predicts for your target, escalate if `verify` fails.
    - "cascade": always start at `start`, climb one level at a time while `verify` fails.
    - "fixed":   always use `start`, never escalate. Useful as a baseline.
    """

    def __init__(self, client, model: str = DEFAULT_MODEL, router: Optional[Router] = None, target: float = 0.85,
                 policy: str = "router", start: str = "low", max_effort: str = "max", max_attempts: int = 3,
                 fallback: str = "medium", max_tokens_cap: int = 64000, prices: Optional[tuple] = None,
                 on_attempt: Optional[Callable[[Attempt], None]] = None, long_think_tokens: Optional[int] = None,
                 long_think_to: str = "high"):
        if policy not in ("router", "cascade", "fixed"):
            raise ValueError("policy must be router, cascade or fixed")
        for name, lvl in (("start", start), ("max_effort", max_effort), ("fallback", fallback),
                          ("long_think_to", long_think_to)):
            if lvl not in LEVELS:
                raise ValueError(f"{name} must be one of {LEVELS}")
        self.client, self.model, self.target, self.policy = client, model, target, policy
        self.start, self.max_effort, self.fallback = start, max_effort, fallback
        self.max_attempts, self.max_tokens_cap = max(1, max_attempts), max_tokens_cap
        self.prices, self.on_attempt = prices, on_attempt
        self.long_think_tokens, self.long_think_to = long_think_tokens, long_think_to
        self.router = router if router is not None else (Router.load() if policy == "router" else None)

    @staticmethod
    def _last_user_text(messages) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content", "")
                if isinstance(c, list):  # content parts
                    return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
                return str(c)
        return ""

    def pick(self, messages) -> str:
        if self.policy != "router":
            return self.start
        choice = self.router.explain(self._last_user_text(messages), self.target)
        lvl = choice.effort if choice.in_domain else self.fallback
        return LEVELS[min(LEVELS.index(lvl), LEVELS.index(self.max_effort))]

    def create(self, messages, verify: Optional[Callable[[str], bool]] = None, escalate: bool = True,
               effort: Optional[str] = None, **kwargs) -> DialResult:
        """effort=... skips the router for this call. Extra kwargs go straight to chat.completions.create."""
        if "reasoning_effort" in kwargs:
            raise ValueError("pass effort=... instead of reasoning_effort; BeamDial sets reasoning_effort itself")
        routed = effort or self.pick(messages)
        cur = routed
        top = LEVELS.index(self.max_effort)
        attempts, resp = [], None
        for _ in range(self.max_attempts):
            resp = self.client.chat.completions.create(model=self.model, messages=messages,
                                                       reasoning_effort=cur, **kwargs)
            ch = resp.choices[0]
            content = ch.message.content
            usage = getattr(resp, "usage", None)
            details = getattr(usage, "completion_tokens_details", None) if usage else None
            has_tools = bool(getattr(ch.message, "tool_calls", None))

            passed = None
            if ch.finish_reason == "length":
                outcome = "truncated"
            elif has_tools:
                outcome = "tool_call"
            elif not content:
                outcome = "empty"
            else:
                if verify is not None:
                    try:
                        passed = bool(verify(content))
                    except Exception:
                        passed = False
                outcome = "ok" if passed is not False else "verify_failed"

            att = Attempt(cur, content, ch.finish_reason, getattr(usage, "prompt_tokens", 0) or 0,
                          getattr(usage, "completion_tokens", 0) or 0,
                          getattr(details, "reasoning_tokens", None) if details else None,
                          kwargs.get("max_completion_tokens"), passed, outcome)
            attempts.append(att)
            if self.on_attempt:
                self.on_attempt(att)

            # Check-free signal: a cheap level that thought for a long time probably met a hard prompt.
            thought = att.reasoning_tokens if att.reasoning_tokens is not None else att.completion_tokens
            if (outcome == "ok" and verify is None and escalate and self.policy != "fixed"
                    and self.long_think_tokens and thought > self.long_think_tokens
                    and LEVELS.index(cur) < LEVELS.index(self.long_think_to) <= top
                    and len(attempts) < self.max_attempts):
                att.outcome = "long_think"
                cur = self.long_think_to
                continue
            if outcome in ("ok", "tool_call") or not escalate or self.policy == "fixed":
                break
            if outcome == "truncated":
                budget = kwargs.get("max_completion_tokens")
                if budget and budget < self.max_tokens_cap:
                    kwargs["max_completion_tokens"] = min(budget * 2, self.max_tokens_cap)
                    continue  # same level, bigger budget
                break  # no room to grow: return what we have, flagged as truncated
            i = LEVELS.index(cur)
            if i >= top:
                break
            cur = LEVELS[i + 1]

        last = attempts[-1]
        return DialResult(resp, last.content, last.effort, routed, attempts, self.prices)
