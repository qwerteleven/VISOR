import json
import pytest
import object_segmentation.sam_api as webcam_api


@pytest.fixture(autouse=True)
def reset_global_state():
    webcam_api.ref_points = []
    webcam_api.skip_next_touch_end = False
    yield
    webcam_api.ref_points = []
    webcam_api.skip_next_touch_end = False


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    return TestClient(webcam_api.app)


def test_touch_start_sets_single_reference_point(client):
    with client.websocket_connect("/ws/overlay") as ws:
        ws.send_text(json.dumps({"type": "touch_start", "x": 10, "y": 20}))
        ack = ws.receive_text()

    assert json.loads(ack) == {"ok": True}
    assert webcam_api.ref_points == [(10, 20)]


def test_touch_start_then_touch_end_forms_a_pair(client):
    with client.websocket_connect("/ws/overlay") as ws:
        ws.send_text(json.dumps({"type": "touch_start", "x": 1, "y": 1}))
        ws.receive_text()
        ws.send_text(json.dumps({"type": "touch_end", "x": 2, "y": 2}))
        ack = ws.receive_text()

    assert json.loads(ack) == {"ok": True}
    assert webcam_api.ref_points == [(1, 1), (2, 2)]


def test_new_touch_start_after_complete_pair_resets_and_sets_skip_flag(client):
    with client.websocket_connect("/ws/overlay") as ws:
        ws.send_text(json.dumps({"type": "touch_start", "x": 1, "y": 1}))
        ws.receive_text()
        ws.send_text(json.dumps({"type": "touch_end", "x": 2, "y": 2}))
        ws.receive_text()

        ws.send_text(json.dumps({"type": "touch_start", "x": 3, "y": 3}))
        ack = ws.receive_text()

    assert json.loads(ack) == {"ok": True}
    assert webcam_api.ref_points == [(3, 3)]
    assert webcam_api.skip_next_touch_end is True


def test_skipped_touch_end_does_not_modify_ref_points_but_still_sends_ack(client):
    with client.websocket_connect("/ws/overlay") as ws:
        ws.send_text(json.dumps({"type": "touch_start", "x": 1, "y": 1}))
        ws.receive_text()
        ws.send_text(json.dumps({"type": "touch_end", "x": 2, "y": 2}))
        ws.receive_text()
        ws.send_text(json.dumps({"type": "touch_start", "x": 3, "y": 3}))
        ws.receive_text()

        ws.send_text(json.dumps({"type": "touch_end", "x": 4, "y": 4}))
        ack = ws.receive_text()

    assert json.loads(ack) == {"ok": True}
    assert webcam_api.ref_points == [(3, 3)]
    assert webcam_api.skip_next_touch_end is False


def test_skip_flag_only_applies_once(client):
    with client.websocket_connect("/ws/overlay") as ws:
        ws.send_text(json.dumps({"type": "touch_start", "x": 1, "y": 1}))
        ws.receive_text()
        ws.send_text(json.dumps({"type": "touch_end", "x": 2, "y": 2}))
        ws.receive_text()
        ws.send_text(json.dumps({"type": "touch_start", "x": 3, "y": 3}))
        ws.receive_text()

        ws.send_text(json.dumps({"type": "touch_end", "x": 4, "y": 4}))
        ws.send_text(json.dumps({"type": "touch_end", "x": 5, "y": 5}))
        ack = ws.receive_text()

    assert json.loads(ack) == {"ok": True}
    assert webcam_api.ref_points == [(3, 3), (5, 5)]


def test_unknown_message_type_still_gets_acked(client):
    with client.websocket_connect("/ws/overlay") as ws:
        ws.send_text(json.dumps({"type": "something_else", "x": 1, "y": 1}))
        ack = ws.receive_text()

    assert json.loads(ack) == {"ok": True}
    assert webcam_api.ref_points == []


def test_invalid_json_is_handled_without_crashing_the_server(client):
    with client.websocket_connect("/ws/overlay") as ws:
        ws.send_text("not valid json{{{")

    with client.websocket_connect("/ws/overlay") as ws2:
        ws2.send_text(json.dumps({"type": "touch_start", "x": 7, "y": 7}))
        ack = ws2.receive_text()
    assert json.loads(ack) == {"ok": True}
