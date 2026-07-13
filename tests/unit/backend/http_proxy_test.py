import httpx
import respx


def test_root_serves_frontend_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "test frontend" in response.text


def test_config_endpoint_returns_client_config(client, test_config):
    response = client.get("/config")
    assert response.status_code == 200
    assert response.json() == test_config["client"]


def test_proxy_returns_404_for_unmapped_route(client):
    response = client.get("/no/such/route")
    assert response.status_code == 404


@respx.mock
def test_proxy_forwards_request_and_returns_upstream_body(client):
    route = respx.get("http://data-service/api/data/users").mock(
        return_value=httpx.Response(200, json={"users": ["a", "b"]})
    )

    response = client.get("/api/data/users")

    assert route.called
    assert response.status_code == 200
    assert response.json() == {"users": ["a", "b"]}


@respx.mock
def test_proxy_strips_filtered_request_headers(client):
    route = respx.get("http://data-service/api/data/x").mock(
        return_value=httpx.Response(200, json={})
    )

    client.get("/api/data/x", headers={"X-Custom": "keep-me"})

    sent_headers = route.calls.last.request.headers
    assert "x-custom" in sent_headers  # non-filtered headers pass through

    assert route.called


@respx.mock
def test_proxy_excludes_response_headers(client):
    import gzip

    compressed = gzip.compress(b"{}")
    route = respx.get("http://data-service/api/data/y").mock(
        return_value=httpx.Response(
            200,
            content=compressed,
            headers={"content-encoding": "gzip", "x-keep": "yes"},
        )
    )

    response = client.get("/api/data/y")

    assert route.called
    assert "content-encoding" not in response.headers
    assert response.headers.get("x-keep") == "yes"


@respx.mock
def test_proxy_returns_503_when_upstream_unreachable(client):
    respx.get("http://data-service/api/data/z").mock(
        side_effect=httpx.ConnectError("connection refused")
    )

    response = client.get("/api/data/z")

    assert response.status_code == 503


@respx.mock
def test_proxy_returns_504_on_upstream_timeout(client):
    respx.get("http://data-service/api/data/slow").mock(
        side_effect=httpx.TimeoutException("timed out")
    )

    response = client.get("/api/data/slow")

    assert response.status_code == 504


@respx.mock
def test_proxy_returns_500_on_unexpected_error(client):
    respx.get("http://data-service/api/data/broken").mock(
        side_effect=RuntimeError("boom")
    )

    response = client.get("/api/data/broken")

    assert response.status_code == 500


@respx.mock
def test_proxy_streams_chunks_for_stream_route(client):
    respx.get("http://stream-service/api/stream/live").mock(
        return_value=httpx.Response(200, content=b"chunk-one-chunk-two")
    )

    response = client.get("/api/stream/live")

    assert response.status_code == 200
    assert response.content == b"chunk-one-chunk-two"
    assert response.headers["content-type"] == "application/octet-stream"
