import asyncio
import json
import time
import cv2
import sys
import os
import threading
import traceback
import logging
from contextlib import asynccontextmanager
from PIL import Image, ImageDraw
import io
import numpy as np
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse


root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

from demo_trt_webcam import sam3_model
from utils.io import get_config, set_logger

set_logger("../logs", os.path.basename(sys.argv[0]))


overlay_lock = asyncio.Lock()

sam_quality = get_config("config.json", "sam_streaming_quality") 
sam_timeout = get_config("config.json", "sam_api_timeouts") 
config = get_config("config.json", "demo_trt_webcam") 
overlay_config = get_config("../config.json", "streaming_overlay")
input_source = get_config("../config.json", "input_source")
streaming_mediatype = get_config("../config.json", "streaming_mediatype")
endpoints = get_config("../config.json", "endpoints")
ml_model = sam3_model(config["engine_file_path"], config, overlay_config)
ml_model.load()


last_frame_time = 0  
latest_frame: np.ndarray | None = None
frame_lock = threading.Lock()
ref_points: List = []
skip_next_touch_end: bool = False

def rtsp_reader(rtsp_url: str) -> None:
    """
    
        consume frame from source with opencv

    Args:
        rtsp_url (str): streaming source 
    """    
    global latest_frame
    while True:
        try:
            print(f"Connecting to RTSP: {rtsp_url}")
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not cap.isOpened():
                print("RTSP not available, retrying in 5s...")
                time.sleep(sam_timeout["connect_rtsp"])
                continue

            last_frame_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    if time.time() - last_frame_time > sam_timeout["stale_connection"]:
                        print("RTSP stale — reconnecting...")
                        break
                    time.sleep(sam_timeout["read_frame"])
                    continue

                last_frame_time = time.time()
                with frame_lock:
                    latest_frame = frame.copy()

            cap.release() 
            print(f"RTSP disconnected, reconnecting in {sam_timeout['disconnect_retry']}s...")
            time.sleep(sam_timeout['disconnect_retry'])

        except Exception as e:
            msg = f"RTSP error: {e}, retrying in {sam_timeout['rtsp_retry']}s..."
            logging.error(msg)
            print(msg)
            print(traceback.format_exc())
            time.sleep(sam_timeout['rtsp_retry'])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    
        thread for reading the streaming

    Args:
        app (FastAPI): API object
    """  
    rtsp_url = input_source
    t = threading.Thread(target=rtsp_reader, args=(rtsp_url,), daemon=True)
    t.start()
    yield

app = FastAPI(lifespan=lifespan)


def apply_overlay(frame: np.ndarray) -> bytes:
    """
        apply ml model inference result, final result be conveted to PIL format, a later past to byte_buffer

    Args:
        frame (np.ndarray): image cv2 format

    Returns:
        bytes: buffer with image + overlay
    """    

    global ref_points
    
    ml_model.update_attention_region(ref_points, frame.shape)
    output_image = ml_model(frame)

    img = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    buf = io.BytesIO()
    img.save(buf, format=sam_quality["format"], quality=sam_quality["quality"])
    return buf.getvalue()


async def frame_generator():
    """
        Generates the frames + overlay for HTTPS web

    Yields:
        bytes: streaming frames sended to web
    """    

    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            await asyncio.sleep(sam_timeout["frame_generator"])
            continue

        jpeg = apply_overlay(frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        )
        await asyncio.sleep(1 / sam_quality["fps"])


@app.get(endpoints["stream"])
async def stream() -> StreamingResponse:
    """
    
        Gives to HTTPS web the streaming channel output

    Returns:
        StreamingResponse: standar responses of Fast API
    """    
    return StreamingResponse(
        frame_generator(),
        media_type=streaming_mediatype,
    )


@app.websocket(endpoints["overlay"])
async def overlay_ws(ws: WebSocket):
    """
    
        Manage the comunication between user and ML model

    Args:
        ws (WebSocket): conection between user - API
    """    
    await ws.accept()

    try:
        while True:
            global ref_points
            global skip_next_touch_end
            data = await ws.receive_text()
            msg = json.loads(data)
            async with overlay_lock:

                if msg["type"] == "touch_start":
                    if len(ref_points) >= 2:
                        skip_next_touch_end = True

                    ref_points = [(msg["x"], msg["y"])]

                if msg["type"] == "touch_end":
                    if skip_next_touch_end:
                        skip_next_touch_end = False
                        continue
                    ref_points += [(msg["x"], msg["y"])]

            await ws.send_text(json.dumps({"ok": True}))

    except WebSocketDisconnect:
        msg = f"Web Socket Disconnect"
        logging.error(msg)
        print(msg)
    

