import asyncio
import logging
from typing import Dict, Optional, Tuple

import httpx
import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles


def create_app(config: Dict, static_directory: Optional[str] = None) -> FastAPI:
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config["CORS_ALLOW"]["allow_origins"],
        allow_methods=config["CORS_ALLOW"]["allow_methods"],
        allow_headers=config["CORS_ALLOW"]["allow_headers"],
    )

    if static_directory is not None:
        app.mount(
            config["endpoints"]["static"],
            StaticFiles(directory=static_directory),
            name="static",
        )

    @app.get(config["endpoints"]["root"])
    async def root() -> FileResponse:
        return FileResponse(config["frontend_index_file"])

    @app.get(config["endpoints"]["config"])
    async def get_client_config() -> Dict:
        try:
            return config["client"]
        except KeyError:
            logging.error("key client is not in general config")
            return {}

    def get_target(path: str) -> Tuple[Optional[str], Optional[str]]:
        for service in config["services"]:
            if path.startswith(service["route"]):
                return service["target"], service["route"]
        return None, None

    async def _stream_generator(
        request: Request, timeout: httpx.Timeout, url: str, headers: Dict
    ):
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                method=request.method,
                url=url,
                headers=headers,
                params=dict(request.query_params),
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

    async def _client_to_backend(websocket: WebSocket, backend) -> None:
        try:
            while True:
                try:
                    data = await websocket.receive_text()
                    await backend.send(data)
                except WebSocketDisconnect:
                    break
        except Exception as e:
            logging.error(f"Unexpected error websocket client - backend, error: {e}")

    async def _backend_to_client(websocket: WebSocket, backend) -> None:
        try:
            async for message in backend:
                try:
                    await websocket.send_text(message)
                except WebSocketDisconnect:
                    break
        except Exception as e:
            logging.error(f"Unexpected error websocket backend - client, error: {e}")

    @app.websocket(config["endpoints"]["overlay"])
    async def websocket_proxy(websocket: WebSocket):
        await websocket.accept()
        try:
            async with websockets.connect(
                f"ws://localhost:{config['sam_port']}{config['endpoints']['overlay']}",
                open_timeout=config["websocket_connection"]["open_timeout"],
                ping_interval=config["websocket_connection"]["ping_interval"],
                ping_timeout=config["websocket_connection"]["ping_timeout"],
            ) as backend:
                await asyncio.gather(
                    _client_to_backend(websocket, backend),
                    _backend_to_client(websocket, backend),
                )
        except Exception as e:
            logging.error(f"WS proxy error: {e}")
        finally:
            try:
                await websocket.close()
            except Exception as e:
                logging.error(f"WS proxy error closing connection: {e}")

    @app.api_route("/{path:path}", methods=config["allow_methods"]["proxy"])
    async def proxy(request: Request, path: str):
        full_path = f"/{path}"
        target, route = get_target(full_path)

        if not target:
            return Response(content="Not found", status_code=404)

        url = f"{target}{full_path}"
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in config["filter_headers"]
        }

        try:
            if route == config["endpoints"]["stream"]:
                timeout = httpx.Timeout(config["streaming_timeout"], read=None)
                return StreamingResponse(
                    _stream_generator(request, timeout, url, headers),
                    media_type=config["streaming_mediatype"],
                )

            timeout = httpx.Timeout(
                config["client_timeouts"]["time"],
                read=config["client_timeouts"]["read"],
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
                if k.lower() not in config["exclude_headers"]
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
            logging.error(f"Unhandled exception: {e}")
            return Response(content="Internal Server Error", status_code=500)

    return app
