import asyncio
import json
import time
import cv2
import sys
import os
import threading
from contextlib import asynccontextmanager
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse


root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

from demo_trt_webcam import sam3_model
from utils.io import get_config, set_logger


overlay_lock = asyncio.Lock()


config = get_config("config.json", "demo_trt_webcam") 
overlay_config = get_config("../config.json", "streaming_overlay")
output_config = get_config("config.json", "output_vidgear")
input_config = get_config("config.json", "input_vidgear")
output_source = get_config("../config.json", "output_source")
input_source = get_config("../config.json", "input_source")
ml_model = sam3_model(config["engine_file_path"], config, overlay_config)
ml_model.load()



latest_frame: np.ndarray | None = None
frame_lock = threading.Lock()
ref_points: List = []
skip_next_touch_end: bool = False


def rtsp_reader(rtsp_url: str) -> None:
    """
        read frame by frame a RTSP streaming, if the streamign con not be connect try to reconect infinitly
        THIS IS ONLY FOR TESTING, ALL FUNCTIONS MUST HAVE A FINAL ITERATION

    Args:
        rtsp_url (str): path to rtsp streaming
    """    
    global latest_frame
    while True:
        try:
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                time.sleep(5)
                continue

            while True:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reconnect / loop
                    continue
                with frame_lock:
                    latest_frame = frame.copy()

        except Exception as e:
            print("can not read the rstp streaming, retrying in 5 seconds")
          

@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    """
    
        thead for reading the streaming

    Args:
        app (FastAPI): API obaject
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
    if not state["show"]:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        global ref_points
        
        ml_model.update_attention_region(ref_points, frame.shape)
        output_image = ml_model(frame)

        img = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


async def frame_generator():
    """
        Generates the frames + overlay for HTTPS web, 


    Yields:
        bytes: streaming frames sended to web
    """    
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            await asyncio.sleep(0.05)
            continue

        jpeg = apply_overlay(frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        )
        await asyncio.sleep(1 / 30)  # ~30 fps

@app.get("/stream")
async def stream() -> fastapi.responses.StreamingResponse:
    """
    
        Gives to HTTPS web the streaming channel output

    Returns:
        StreamingResponse: standar responses of Fast API
    """    
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/overlay")
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
        pass

