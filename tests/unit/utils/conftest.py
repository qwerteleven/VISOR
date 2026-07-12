import pytest

import utils.wrappers as timing


@pytest.fixture
def fake_clock(monkeypatch):
    class FakeClock:
        def __init__(self, step=0.01):
            self._now = 0.0
            self.step = step

        def time(self):
            value = self._now
            self._now += self.step
            return value

    clock = FakeClock()
    monkeypatch.setattr(timing.time, "time", clock.time)
    return clock
