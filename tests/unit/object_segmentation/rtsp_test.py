import threading
import time
import numpy as np
import pytest

import object_segmentation.sam_api as webcam_api


class FakeVideoCapture:
    def __init__(self, url, frames_to_yield=None, opens_successfully=True):
        self.url = url
        self._frames = list(frames_to_yield or [])
        self._opens_successfully = opens_successfully
        self.released = False
        self.read_calls = 0

    def isOpened(self):
        return self._opens_successfully

    def set(self, *args, **kwargs):
        pass

    def read(self):
        self.read_calls += 1
        if self._frames:
            frame = self._frames.pop(0)
            if frame is None:
                return False, None
            return True, frame
        return False, None

    def release(self):
        self.released = True


@pytest.fixture(autouse=True)
def reset_frame_state():
    with webcam_api.frame_lock:
        webcam_api.latest_frame = None
    yield
    with webcam_api.frame_lock:
        webcam_api.latest_frame = None


def run_reader_briefly(fake_capture_factory, duration=0.2):
    thread = threading.Thread(
        target=webcam_api.rtsp_reader, args=("fake://url",), daemon=True
    )
    thread.start()
    time.sleep(duration)
    return thread


def test_rtsp_reader_sets_latest_frame_from_a_successful_read(monkeypatch):
    real_frame = np.full((4, 4, 3), 42, dtype=np.uint8)
    monkeypatch.setattr(
        webcam_api.cv2,
        "VideoCapture",
        lambda url: FakeVideoCapture(url, frames_to_yield=[real_frame]),
    )

    run_reader_briefly(None, duration=0.1)

    with webcam_api.frame_lock:
        got = webcam_api.latest_frame

    assert got is not None
    np.testing.assert_array_equal(got, real_frame)


def test_rtsp_reader_stores_a_copy_not_a_live_reference(monkeypatch):
    real_frame = np.full((4, 4, 3), 10, dtype=np.uint8)
    monkeypatch.setattr(
        webcam_api.cv2,
        "VideoCapture",
        lambda url: FakeVideoCapture(url, frames_to_yield=[real_frame]),
    )

    run_reader_briefly(None, duration=0.1)

    real_frame[:] = 99

    with webcam_api.frame_lock:
        got = webcam_api.latest_frame

    assert got is not None
    assert not np.array_equal(got, real_frame)


def test_rtsp_reader_retries_when_capture_fails_to_open(monkeypatch):
    open_attempts = []

    def fake_video_capture(url):
        open_attempts.append(url)
        return FakeVideoCapture(url, opens_successfully=False)

    monkeypatch.setattr(webcam_api.cv2, "VideoCapture", fake_video_capture)

    thread = run_reader_briefly(None, duration=0.15)

    assert len(open_attempts) > 1
    assert thread.is_alive()


def test_rtsp_reader_survives_an_exception_from_video_capture(monkeypatch):
    call_count = {"n": 0}
    real_frame = np.full((3, 3, 3), 7, dtype=np.uint8)

    def flaky_video_capture(url):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated transient driver failure")
        return FakeVideoCapture(url, frames_to_yield=[real_frame])

    monkeypatch.setattr(webcam_api.cv2, "VideoCapture", flaky_video_capture)

    run_reader_briefly(None, duration=0.15)

    with webcam_api.frame_lock:
        got = webcam_api.latest_frame

    assert got is not None
    np.testing.assert_array_equal(got, real_frame)
