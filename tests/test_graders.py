"""Regression tests for the graders. The first full run under-counted math because math-verify
failed inside worker threads and the error was counted as a wrong answer."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
pytest.importorskip("math_verify")
from graders import grade_math  # noqa: E402

EQUIVALENT = [
    ("\\frac14", "\\frac{1}{4}"), ("15 \\pi", "15\\pi"), ("36x \\sqrt{35x}", "36x\\sqrt{35x}"),
    ("(4x - 7)", "4x-7"), ("\\sqrt{101} - \\sqrt{2} + 9", "9+\\sqrt{101}-\\sqrt{2}"),
    ("\\frac{2}{7} \\sqrt{53}", "\\dfrac{2\\sqrt{53}}{7}"), ("\\frac{-1}{1997}", "-\\dfrac{1}{1997}"),
    ("073", "73"), ("0.5", "\\frac{1}{2}"),
]


@pytest.mark.parametrize("gold,pred", EQUIVALENT)
def test_equivalent_forms_are_correct(gold, pred):
    assert grade_math(f"so the answer is \\boxed{{{pred}}}", gold)["correct"]


def test_wrong_answer_is_wrong():
    assert not grade_math("\\boxed{6}", "3")["correct"]


def test_refuses_worker_threads_instead_of_silently_failing():
    with ThreadPoolExecutor(1) as ex:
        with pytest.raises(RuntimeError):
            ex.submit(grade_math, "\\boxed{\\frac{1}{4}}", "\\frac14").result()
