from contextlib import asynccontextmanager
import pytest
import backend.proxy as proxy


class FakeBackendConnection:
    def __init__(self, messages_to_send=None):
        self.sent_messages = []
        self._messages_to_send = list(messages_to_send or [])

    async def send(self, data):
        self.sent_messages.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages_to_send:
            raise StopAsyncIteration
        return self._messages_to_send.pop(0)


@pytest.fixture
def fake_backend():
    return FakeBackendConnection(messages_to_send=["frame-1-result", "frame-2-result"])


@pytest.fixture
def patch_backend_connect(monkeypatch, fake_backend):
    @asynccontextmanager
    async def fake_connect(*args, **kwargs):
        yield fake_backend

    monkeypatch.setattr(proxy.websockets, "connect", fake_connect)
    return fake_backend


def test_websocket_proxy_forwards_backend_messages_to_client(
    client, patch_backend_connect
):
    with client.websocket_connect("/ws/overlay") as websocket:
        assert websocket.receive_text() == "frame-1-result"
        assert websocket.receive_text() == "frame-2-result"


def test_websocket_proxy_forwards_client_messages_to_backend(
    client, patch_backend_connect
):
    with client.websocket_connect("/ws/overlay") as websocket:
        websocket.send_text("client-command")
        websocket.receive_text()

    assert "client-command" in patch_backend_connect.sent_messages


def test_client_to_backend_outer_except_covers_send_failure(monkeypatch, client):
    """backend.send() raising something other than WebSocketDisconnect hits
    the outer `except Exception` in _client_to_backend (lines 70-71)."""

    class BrokenSendBackend(FakeBackendConnection):
        async def send(self, data):
            raise RuntimeError("backend send exploded")

    broken_backend = BrokenSendBackend(messages_to_send=[])

    @asynccontextmanager
    async def fake_connect(*args, **kwargs):
        yield broken_backend

    monkeypatch.setattr(proxy.websockets, "connect", fake_connect)

    with client.websocket_connect("/ws/overlay") as websocket:
        websocket.send_text("this will fail to forward")


def test_backend_to_client_outer_except_covers_iteration_failure(monkeypatch, client):
    """Iterating the backend raising something other than StopAsyncIteration
    hits the outer `except Exception` in _backend_to_client (lines 80-81)."""

    class BrokenIterationBackend(FakeBackendConnection):
        async def __anext__(self):
            raise RuntimeError("backend iteration exploded")

    broken_backend = BrokenIterationBackend()

    @asynccontextmanager
    async def fake_connect(*args, **kwargs):
        yield broken_backend

    monkeypatch.setattr(proxy.websockets, "connect", fake_connect)

    with client.websocket_connect("/ws/overlay"):
        pass


def test_websocket_proxy_outer_except_covers_connect_failure(monkeypatch, client):
    """websockets.connect() itself failing (e.g. backend service down) hits
    the outer `except Exception` in websocket_proxy (lines 97-98)."""

    @asynccontextmanager
    async def failing_connect(*args, **kwargs):
        raise ConnectionRefusedError("no backend listening")

    monkeypatch.setattr(proxy.websockets, "connect", failing_connect)

    with client.websocket_connect("/ws/overlay"):
        pass


def test_websocket_proxy_covers_close_failure_in_finally(monkeypatch, client):
    """websocket.close() itself raising hits the except inside the finally
    block (lines 102-103). We patch WebSocket.close at the class level since
    that's the object FastAPI hands the handler, not something our fixtures
    construct directly."""
    from starlette.websockets import WebSocket

    empty_backend = FakeBackendConnection(messages_to_send=[])  # ends immediately

    @asynccontextmanager
    async def fake_connect(*args, **kwargs):
        yield empty_backend

    monkeypatch.setattr(proxy.websockets, "connect", fake_connect)

    async def broken_close(self, *args, **kwargs):
        raise RuntimeError("close exploded")

    monkeypatch.setattr(WebSocket, "close", broken_close)

    with client.websocket_connect("/ws/overlay"):
        pass
