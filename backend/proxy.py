import os
import sys
import asyncio
import logging
import traceback
import httpx
import uvicorn
import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse

from typing import Dict, Tuple


root_folder = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(root_folder)


from utils.io import set_logger, load_config  # noqa: E402

set_logger("../logs", os.path.basename(sys.argv[0]))
config_cache = load_config("../config.json")
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=config_cache["CORS_ALLOW"]["allow_origins"],
    allow_methods=config_cache["CORS_ALLOW"]["allow_methods"],
    allow_headers=config_cache["CORS_ALLOW"]["allow_headers"],
)


app.mount(
    config_cache["endpoints"]["static"],
    StaticFiles(directory="../frontend"),
    name="static",
)


@app.get(config_cache["endpoints"]["root"])
async def root() -> FileResponse:
    """

        give the init of the web

    Returns:
        FileResponse: index.html of the  web
    """

    response = FileResponse(config_cache["frontend_index_file"])

    assert response is not None

    return response


@app.get(config_cache["endpoints"]["config"])
async def get_client_config() -> Dict:
    """

        give to web client internal allow routes

    Returns:
        dict: routes to call services
    """
    try:
        cfg = config_cache["client"]
    except KeyError:
        msg = "key client is not in general config"
        logging.error(msg)
        print(msg)
        print(traceback.format_exc())

    return cfg


async def _client_to_backend(websocket: WebSocket, backend: websockets):
    """

        connection between client to backend

    Args:
        websocket WebSocket: channel input
        backend websockets.connect: channel output
    """

    assert websocket is not None
    assert backend is not None

    try:
        while True:
            try:
                data = await websocket.receive_text()
                await backend.send(data)
            except WebSocketDisconnect:
                msg = "Web Socket Disconnect"
                logging.error(msg)
                print(msg)
                break

    except Exception as e:
        msg = f"Unexpected error websocket client - backend, error: {e}"
        logging.error(msg)
        print(msg)
        print(traceback.format_exc())


async def _backend_to_client(websocket: WebSocket, backend: websockets):
    """

        connection between backend to client

    Args:
        websocket WebSocket: channel output
        backend websockets.connect: channel input
    """

    assert websocket is not None
    assert backend is not None

    try:
        async for message in backend:
            try:
                await websocket.send_text(message)
            except WebSocketDisconnect:
                msg = "Web Socket Disconnect"
                logging.error(msg)
                print(msg)
                break

    except Exception as e:
        msg = f"Unexpected error websocket backend - client, error: {e}"
        logging.error(msg)
        print(msg)
        print(traceback.format_exc())


@app.websocket(config_cache["endpoints"]["overlay"])
async def websocket_proxy(websocket: WebSocket):
    """

        connect websocket comunication between internal streaming service
        and web client

    Args:
        websocket (WebSocket): websocket object
    """

    assert websocket is not None

    await websocket.accept()

    try:
        async with websockets.connect(
            f"ws://localhost:{config_cache['sam_port']}{config_cache['endpoints']['overlay']}",
            open_timeout=config_cache["websocket_connection"]["open_timeout"],
            ping_interval=config_cache["websocket_connection"]["ping_interval"],
            ping_timeout=config_cache["websocket_connection"]["ping_timeout"],
        ) as backend:
            await asyncio.gather(
                _client_to_backend(websocket, backend),
                _backend_to_client(websocket, backend),
            )

    except Exception as e:
        msg = f"WS proxy error: {e}"
        logging.error(msg)
        print(msg)
        print(traceback.format_exc())
    finally:
        try:
            await websocket.close()
        except Exception as e:
            msg = f"WS proxy error clossing connection: {e}"
            logging.error(msg)
            print(msg)
            print(traceback.format_exc())


def get_target(path: str) -> Tuple[str, str | None, None]:
    """

        get the corresponding endpoint for the request

    Args:
        path (str): request path

    Returns:
        Tuple[None, None]: default response
    """

    for service in config_cache["services"]:
        if path.startswith(service["route"]):
            return service["target"], service["route"]

    return None, None


async def _stream_generator(
    request: Request, timeout: httpx.Timeout, url: str, headers: Dict
):
    """

        generates consumable chuck of straming data for the web streaming

    Args:
        request (Request): client request
        timeout (httpx.Timeout): timeout for the connection with client
        url (str): url of site for data offering
        headers (Dict): data for handshake with client

    Yields:
        Iterator[AsyncIterator[bytes]]: chuck of bytes of the streaming
    """

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            method=request.method,
            url=url,
            headers=headers,
            params=dict(request.query_params),
        ) as response:
            async for chunk in response.aiter_bytes():
                yield chunk


@app.api_route("/{path:path}", methods=config_cache["allow_methods"]["proxy"])
async def proxy(request: Request, path: str):
    """

        creates the connections between internal services and the client

    Args:
        request (Request): client request
        path (str): endpoint of the request

    Returns:
        Response: respose of server to client
    """

    assert request is not None

    full_path = f"/{path}"
    target, route = get_target(full_path)

    if not target:
        return Response(content="Not found", status_code=404)

    url = f"{target}{full_path}"
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in config_cache["filter_headers"]
    }

    try:
        if route == config_cache["endpoints"]["stream"]:
            timeout = httpx.Timeout(config_cache["streaming_timeout"], read=None)
            return StreamingResponse(
                _stream_generator(request, timeout, url, headers),
                media_type=config_cache["streaming_mediatype"],
            )

        # Default connection to client
        timeout = httpx.Timeout(
            config_cache["client_timeouts"]["time"],
            read=config_cache["client_timeouts"]["read"],
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                content=await request.body(),
                params=dict(request.query_params),
            )

        response_headers = {
            k: v
            for k, v in response.headers.items()
            if k.lower() not in config_cache["exclude_headers"]
        }

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
        )

    except httpx.ConnectError:
        return Response(content="Service not Available", status_code=503)
    except httpx.TimeoutException:
        return Response(content="Timeout", status_code=504)
    except Exception as e:
        msg = f"Unhandle exception: {e}"
        logging.error(msg)
        print(msg)
        print(traceback.format_exc())
        return Response(content="Internal Server Error", status_code=500)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config_cache["proxy_config"]["host"],
        port=config_cache["proxy_config"]["port"],
        ssl_keyfile=config_cache["keyfile"],
        ssl_certfile=config_cache["certified"],
        ws=config_cache["proxy_config"]["ws"],
        http=config_cache["proxy_config"]["http"],
        loop=config_cache["proxy_config"]["loop"],
    )
