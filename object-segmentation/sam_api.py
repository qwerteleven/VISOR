
import asyncio
import json
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from collections import deque

app = FastAPI()

# Colas compartidas entre los dos endpoints
frame_queue = asyncio.Queue(maxsize=5)
touch_queue = deque(maxlen=20)   # últimos 20 toques

# Cola de resultados para enviar a la web
result_queue = asyncio.Queue()

@app.websocket("/video")
async def video_ws(ws: WebSocket):
    await ws.accept()
    async for data in ws.iter_bytes():
        # data son los bytes del JPEG
        await frame_queue.put((data, asyncio.get_event_loop().time()))

@app.websocket("/touch")
async def touch_ws(ws: WebSocket):
    await ws.accept()
    async for text in ws.iter_text():
        event = json.loads(text)
        touch_queue.append(event)

@app.websocket("/results")
async def results_ws(ws: WebSocket):
    """La página web se conecta aquí para recibir resultados"""
    await ws.accept()
    while True:
        result = await result_queue.get()
        await ws.send_json(result)

# Tarea de procesado — corre en paralelo
async def procesador():
    while True:
        frame_bytes, frame_ts = await frame_queue.get()

        # Buscar toques cercanos en tiempo (±200ms)
        toques_cercanos = [
            t for t in touch_queue
            if abs(t["ts"] / 1000 - frame_ts) < 0.2
        ]

        # Decodificar frame con OpenCV
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        resultado = {"toques": toques_cercanos, "frame_ts": frame_ts}

        # Si hay toque, analizar la zona tocada
        if toques_cercanos and frame is not None:
            h, w = frame.shape[:2]
            for toque in toques_cercanos:
                px = int(toque["x"] * w)
                py = int(toque["y"] * h)
                # Recortar región 50x50 alrededor del toque
                roi = frame[max(0,py-25):py+25, max(0,px-25):px+25]
                # Aquí aplicas tu procesado: ML, color dominante, OCR…
                resultado["zona_tocada"] = {"x": px, "y": py}

        await result_queue.put(resultado)

@app.on_event("startup")
async def startup():
    asyncio.create_task(procesador())