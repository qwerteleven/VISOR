import logging
import pytest

from utils.wrappers import timer


def test_wrapper_returns_the_function_result(fake_clock):
    @timer
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_wrapper_passes_args_and_kwargs_through(fake_clock):
    calls = []

    @timer
    def record(*args, **kwargs):
        calls.append((args, kwargs))

    record(1, 2, x=3)

    assert calls == [((1, 2), {"x": 3})]


def test_does_not_log_before_100_calls(fake_clock, caplog, capsys):
    @timer
    def noop():
        return None

    caplog.set_level(logging.INFO)
    for _ in range(99):
        noop()

    assert caplog.records == []
    assert capsys.readouterr().out == ""


def test_logs_mean_duration_after_exactly_100_calls(fake_clock, caplog, capsys):
    fake_clock.step = 0.02

    @timer
    def noop():
        return None

    caplog.set_level(logging.INFO)
    for _ in range(100):
        noop()

    expected_mean = fake_clock.step
    out = capsys.readouterr().out
    assert f"Mean process time: {expected_mean}" in out
    assert any(
        f"Mean process time: {expected_mean}" in r.message for r in caplog.records
    )


def test_counter_resets_after_hitting_100(fake_clock, caplog, capsys):
    @timer
    def noop():
        return None

    caplog.set_level(logging.INFO)
    for _ in range(100):
        noop()
    caplog.clear()
    capsys.readouterr()

    noop()

    assert caplog.records == []
    assert capsys.readouterr().out == ""


def test_logs_again_after_a_second_full_100_calls(fake_clock, caplog, capsys):
    @timer
    def noop():
        return None

    caplog.set_level(logging.INFO)

    fake_clock.step = 0.01
    for _ in range(100):
        noop()
    capsys.readouterr()  # drain first batch's output

    fake_clock.step = 0.05
    for _ in range(100):
        noop()

    out = capsys.readouterr().out
    assert f"Mean process time: {fake_clock.step}" in out


def test_each_decorated_function_has_independent_counters(fake_clock, caplog, capsys):
    @timer
    def func_a():
        return "a"

    @timer
    def func_b():
        return "b"

    caplog.set_level(logging.INFO)

    for _ in range(50):
        func_a()

    for _ in range(50):
        func_b()

    assert capsys.readouterr().out == ""

    for _ in range(50):
        func_a()

    out = capsys.readouterr().out
    assert "Mean process time" in out
    assert out.count("Mean process time") == 1


def test_exception_in_wrapped_function_propagates(fake_clock):
    @timer
    def boom():
        raise ValueError("something broke")

    with pytest.raises(ValueError, match="something broke"):
        boom()


def test_exception_in_wrapped_function_does_not_advance_counter(
    fake_clock, caplog, capsys
):
    @timer
    def maybe_fail(should_fail):
        if should_fail:
            raise RuntimeError("boom")
        return "ok"

    caplog.set_level(logging.INFO)

    for _ in range(99):
        maybe_fail(False)

    with pytest.raises(RuntimeError):
        maybe_fail(True)  # must NOT count as call #100

    assert capsys.readouterr().out == ""

    maybe_fail(False)

    out = capsys.readouterr().out
    assert "Mean process time" in out


def test_duration_assertions_hold_for_zero_duration_calls(fake_clock):
    fake_clock.step = 0.0

    @timer
    def instant():
        return "done"

    for _ in range(5):
        assert instant() == "done"
