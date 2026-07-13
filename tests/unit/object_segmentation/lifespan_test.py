import threading
import time
import object_segmentation.sam_api as webcam_api


def test_lifespan_starts_rtsp_reader_as_a_daemon_thread(monkeypatch):
    started_with = {}

    def fake_rtsp_reader(rtsp_url):
        started_with["url"] = rtsp_url
        started_with["thread_is_daemon"] = threading.current_thread().daemon

    monkeypatch.setattr(webcam_api, "rtsp_reader", fake_rtsp_reader)

    from fastapi.testclient import TestClient

    with TestClient(webcam_api.app):
        time.sleep(0.1)

    assert started_with["url"] == webcam_api.input_source
    assert started_with["thread_is_daemon"] is True
