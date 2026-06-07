
import asyncio
import json
import os 
import sys
from fastapi import FastAPI, WebSocket
from collections import deque

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)


from utils.io import get_config, set_logger
config = get_config("config.json", "demo_trt_webcam") 
set_logger("../logs", os.path.basename(sys.argv[0]))


from utils.streaming import streaming_pipeline_vidgear
from demo_trt_webcam import sam3_model

app = FastAPI()

frame_queue = asyncio.Queue(maxsize=5)
touch_queue = deque(maxlen=20)   
result_queue = asyncio.Queue()


@app.websocket("/load")
async def video_ws(ws: WebSocket):
    await ws.accept()
    async for data in ws.iter_bytes():
        
        await frame_queue.put((data, asyncio.get_event_loop().time()))


@app.websocket("/unload")
async def video_ws(ws: WebSocket):
    await ws.accept()
    async for data in ws.iter_bytes():
        
        await frame_queue.put((data, asyncio.get_event_loop().time()))


@app.websocket("/touch")
async def touch_ws(ws: WebSocket):
    await ws.accept()
    async for text in ws.iter_text():
        event = json.loads(text)
        touch_queue.append(event)


async def procesador():
    
    overlay_config = get_config("../config.json", "streaming_overlay")
    output_config = get_config("../config.json", "output_vidgear")
    input_config = get_config("../config.json", "input_vidgear")
    output_source = get_config("../config.json", "output_source")
    input_source = get_config("../config.json", "input_source")
    ml_model = sam3_model(config["engine_file_path"], config, overlay_config)
    ml_model.load()

    streaming_pipeline_vidgear(config, ml_model, input_source, input_config, output_source, output_config)



async def lifespan(app: FastAPI):
    
    asyncio.create_task(procesador())
    yield



app = FastAPI(docs_url=None, redoc_url=None, lifespan=lifespan)


