import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
from backend.proxy import create_app  # noqa: E402


@pytest.fixture
def test_config(tmp_path):
    """

    A minimal but complete config covering everything create_app reads.

    """
    index_file = tmp_path / "index.html"
    index_file.write_text("<html>test frontend</html>")

    return {
        "CORS_ALLOW": {
            "allow_origins": ["*"],
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        },
        "endpoints": {
            "static": "/static",
            "root": "/",
            "config": "/config",
            "overlay": "/ws/overlay",
            "stream": "/api/stream",
        },
        "frontend_index_file": str(index_file),
        "client": {"api_base": "/api"},
        "sam_port": 9001,
        "websocket_connection": {
            "open_timeout": 2,
            "ping_interval": 5,
            "ping_timeout": 5,
        },
        "services": [
            {"route": "/api/stream", "target": "http://stream-service"},
            {"route": "/api/data", "target": "http://data-service"},
        ],
        "allow_methods": {"proxy": ["GET", "POST"]},
        "filter_headers": ["host"],
        "exclude_headers": ["content-encoding"],
        "streaming_timeout": 30,
        "streaming_mediatype": "application/octet-stream",
        "client_timeouts": {"time": 5, "read": 5},
    }


@pytest.fixture
def app(test_config):
    return create_app(test_config)


@pytest.fixture
def client(app):
    return TestClient(app)
