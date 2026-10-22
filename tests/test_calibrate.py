"""Calibrate's analysis, tested without an API."""
from beamdial.calibrate import analyze, default_check, mcnemar_p

LV = ["low", "high", "max"]


def res(content, tok, think):
    return {"content": content, "completion_tokens": tok, "reasoning_tokens": think}


def test_default_check_forms():
    assert default_check("so \\boxed{42}", {"answer": "42"})
    assert default_check("The answer is\n42.", {"answer": "42"})
    assert default_check("\\boxed{1,000}", {"answer": "1000"})
    assert not default_check("\\boxed{41}", {"answer": "42"})


def test_mcnemar():
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(10, 0) < 0.01


def test_no_gap_recommends_cheap():
    rows = [{"prompt": f"q{i}", "answer": "1"} for i in range(40)]
    R = {}
    for i in range(40):
        for l, t in zip(LV, (300, 900, 2000)):
            R[(i, l)] = res("\\boxed{1}", t, t - 50)
    rep = analyze(rows, R, LV, default_check)
    assert rep["best"] == "low" and "policy='fixed', start='low'" in rep["setting"]


def test_real_gap_found_and_cheapest_full_accuracy_policy_chosen():
    rows = [{"prompt": f"q{i}", "answer": "1"} for i in range(60)]
    R = {}
    for i in range(60):
        hard = i < 20
        R[(i, "low")] = res("\\boxed{0}" if hard else "\\boxed{1}", 5000 if hard else 300, 4900 if hard else 250)
        R[(i, "high")] = res("\\boxed{1}", 6000 if hard else 600, 5900 if hard else 550)
        R[(i, "max")] = res("\\boxed{1}", 9000 if hard else 900, 8900 if hard else 850)
    rep = analyze(rows, R, LV, default_check)
    assert rep["gap"] > 0.3 and rep["p_value"] < 0.05
    assert rep["policy"]["accuracy"] == 1.0
    # here high is cheap on easy prompts, so always-high beats re-asking: calibrate must say so
    assert "policy='fixed', start='high'" in rep["setting"]


def test_length_signal_wins_when_best_level_is_costly_on_easy_prompts():
    rows = [{"prompt": f"q{i}", "answer": "1"} for i in range(100)]
    R = {}
    for i in range(100):
        hard = i < 10
        R[(i, "low")] = res("\\boxed{0}" if hard else "\\boxed{1}", 5000 if hard else 300, 4900 if hard else 250)
        R[(i, "high")] = res("\\boxed{1}", 6000 if hard else 1200, 5900 if hard else 1150)
        R[(i, "max")] = res("\\boxed{1}", 9000 if hard else 1500, 8900 if hard else 1450)
    rep = analyze(rows, R, LV, default_check)
    assert rep["policy"]["accuracy"] == 1.0
    assert "long_think_tokens=" in rep["setting"] and "long_think_to='high'" in rep["setting"]
