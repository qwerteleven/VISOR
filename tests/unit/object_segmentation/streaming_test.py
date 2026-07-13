import asyncio
import numpy as np
import pytest

import object_segmentation.sam_api as webcam_api
from object_segmentation.sam_api import frame_generator


@pytest.fixture(autouse=True)
def reset_frame_state():
    with webcam_api.frame_lock:
        webcam_api.latest_frame = None
    yield
    with webcam_api.frame_lock:
        webcam_api.latest_frame = None


@pytest.mark.asyncio
async def test_frame_generator_yields_nothing_while_no_frame_available():
    gen = frame_generator()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(gen.__anext__(), timeout=0.05)


@pytest.mark.asyncio
async def test_frame_generator_yields_formatted_multipart_frame():
    with webcam_api.frame_lock:
        webcam_api.latest_frame = np.zeros((10, 10, 3), dtype=np.uint8)

    gen = frame_generator()
    chunk = await asyncio.wait_for(gen.__anext__(), timeout=1.0)

    assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
    assert chunk.endswith(b"\r\n")

    body = chunk[len(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n") : -2]
    assert len(body) > 0


@pytest.mark.asyncio
async def test_frame_generator_copies_frame_not_a_live_reference():
    original = np.zeros((10, 10, 3), dtype=np.uint8)
    with webcam_api.frame_lock:
        webcam_api.latest_frame = original

    webcam_api.ml_model.inference_calls = []
    gen = frame_generator()
    await asyncio.wait_for(gen.__anext__(), timeout=1.0)

    assert webcam_api.ml_model.inference_calls == [(10, 10, 3)]


@pytest.mark.asyncio
async def test_frame_generator_yields_multiple_frames_in_sequence():
    with webcam_api.frame_lock:
        webcam_api.latest_frame = np.zeros((5, 5, 3), dtype=np.uint8)

    gen = frame_generator()
    first = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
    second = await asyncio.wait_for(gen.__anext__(), timeout=1.0)

    assert first.startswith(b"--frame")
    assert second.startswith(b"--frame")


def test_stream_endpoint_returns_streaming_response(monkeypatch):
    from fastapi.testclient import TestClient

    async def fake_finite_generator():
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nFAKE_JPEG_BYTES\r\n"

    monkeypatch.setattr(webcam_api, "frame_generator", fake_finite_generator)

    client = TestClient(webcam_api.app)
    response = client.get("/stream")

    assert response.status_code == 200
    assert "multipart/x-mixed-replace" in response.headers["content-type"]
    assert b"--frame" in response.content
    assert b"Content-Type: image/jpeg" in response.content
    assert b"FAKE_JPEG_BYTES" in response.content
