import pytest
import logging

import utils.wrappers as timing
import utils.io as utils_io


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


@pytest.fixture
def clean_root_logger():
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level

    yield root

    for handler in list(root.handlers):
        if handler not in original_handlers:
            handler.close()
            root.removeHandler(handler)
    root.setLevel(original_level)


@pytest.fixture
def fake_logs_config(monkeypatch):
    def fake_get_config(path, section):
        return {
            "time_format": "%Y%m%d",
            "log_format": "%(asctime)s %(message)s",
            "MAX_LOGS": 5,
        }

    monkeypatch.setattr(utils_io, "get_config", fake_get_config)
    return fake_get_config
