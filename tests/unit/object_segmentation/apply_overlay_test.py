import io

import numpy as np
import pytest
from PIL import Image

import object_segmentation.sam_api as webcam_api
from object_segmentation.sam_api import apply_overlay


@pytest.fixture(autouse=True)
def reset_fake_model_and_globals():
    webcam_api.ml_model.attention_region_calls = []
    webcam_api.ml_model.inference_calls = []
    webcam_api.ml_model.output_override = None
    webcam_api.ref_points = []
    yield


def make_bgr_frame(height=10, width=10):
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_apply_overlay_returns_valid_jpeg_bytes():
    frame = make_bgr_frame()

    result = apply_overlay(frame)
    img = Image.open(io.BytesIO(result))
    img.load()
    assert img.format == "JPEG"


def test_apply_overlay_calls_update_attention_region_with_current_ref_points():
    webcam_api.ref_points = [(1, 1), (2, 2)]
    frame = make_bgr_frame()

    apply_overlay(frame)

    assert webcam_api.ml_model.attention_region_calls == [
        ([(1, 1), (2, 2)], frame.shape)
    ]


def test_apply_overlay_passes_frame_to_model_for_inference():
    frame = make_bgr_frame(height=8, width=8)

    apply_overlay(frame)

    assert webcam_api.ml_model.inference_calls == [(8, 8, 3)]


def test_apply_overlay_uses_models_output_not_original_frame():
    frame = make_bgr_frame(height=4, width=4)
    distinct_output = np.full((4, 4, 3), 255, dtype=np.uint8)
    webcam_api.ml_model.output_override = distinct_output

    result = apply_overlay(frame)

    decoded = np.array(Image.open(io.BytesIO(result)).convert("RGB"))
    assert decoded.mean() > 200


def test_apply_overlay_respects_configured_output_format(monkeypatch):
    monkeypatch.setitem(webcam_api.sam_quality, "format", "PNG")
    frame = make_bgr_frame()

    result = apply_overlay(frame)

    img = Image.open(io.BytesIO(result))
    img.load()
    assert img.format == "PNG"


def test_apply_overlay_returns_placeholder_image_on_encode_failure():
    webcam_api.sam_quality["format"] = "NOT_A_REAL_FORMAT"
    frame = make_bgr_frame()

    result = apply_overlay(frame)

    img = Image.open(io.BytesIO(result))
    img.load()
    assert img.size == (512, 512)

    r, g, b = img.convert("RGB").getpixel((256, 256))
    assert r > 200 and g < 50 and b < 50

    webcam_api.sam_quality["format"] = "JPEG"


def test_apply_overlay_placeholder_save_does_not_raise_unhandled_error():
    webcam_api.sam_quality["format"] = "ALSO_NOT_REAL"
    frame = make_bgr_frame()

    result = apply_overlay(frame)

    assert len(result) > 0
    webcam_api.sam_quality["format"] = "JPEG"
