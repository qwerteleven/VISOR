# VISOR

Real-time object segmentation and natural language description system. A user points their phone camera at an object, draws a bounding box, and receives a live AI-generated segmentation mask with a text description — all in a mobile browser, with no app installation required.

```
Mobile Browser ──WebRTC/WebSocket──▶ MediaMTX ──▶ Python Backend (GPU)
     │                                                      │
     └◀──────────── Processed overlay (WebSocket) ──────────┘
```


<video controls width="100%">
  <source src="https://github.com/qwerteleven/VISOR/edit/master/0626.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
![Video](https://github.com/qwerteleven/VISOR/edit/master/0626.mp4)
![Video](0626.mp4)

---



## Disclaimer

**The codebase, develop scope, is the describe in README.MD file, the other files are strategic thinking, future vision of the project, not the real scope**




---

## Stack

| Layer | Technology |
|---|---|
| Frontend | Mobile web browser (WebRTC, Canvas) |
| Media relay | MediaMTX |
| Backend | Python 3.12 + FastAPI |
| Segmentation | SAM-3 (Meta) via TensorRT |
| Description | Gemma-4 (Google) via Transformers |
| Transport | WebSocket + RTSP/WHIP |
| Inference runtime | TensorRT (FP16), CUDA 12.9 |

---

## Requirements

- Python 3.12.11
- NVIDIA GPU with TensorRT support (tested on RTX 5090)
- CUDA 12.9 + cuDNN
- Linux (required for TensorRT)
- MediaMTX binary (see `rtsp-server/versions.txt`)
- FFmpeg binary (see `rtsp-server/versions.txt`)
- Hugging Face account with access granted for:
  - [`facebook/sam3`](https://huggingface.co/facebook/sam3)
  - [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it)

---

## Installation

### 1. Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Hugging Face authentication

Both models require explicit access approval on their Hugging Face model pages before downloading.

```bash
huggingface-cli login
```

### 3. CUDA & cuDNN

Follow the [NVIDIA CUDA installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/). Verify your setup:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### 4. TLS certificate (required for camera access)

Browsers require HTTPS to access the device camera. Generate a self-signed certificate for development:

```bash
openssl x509 -in cert.csr -out cert.pem -req -signkey key.pem -days 1001
```

> **Note:** Self-signed certificates are for development only. Use a CA-issued certificate in production.

### 5. Configuration

```bash
cp config.json.default config.json
```

Edit `config.json` with your machine's IP, port, and paths:

```json
{
    "certified": "server.pem",
    "keyfile": "key.pem",
    "connection": {
        "host": "<your-machine-ip>",
        "port": 8443
    },
    "client": {
        "fastapi_http": "https://<your-machine-ip>:8443",
        "fastapi_ws":   "wss://<your-machine-ip>:8443",
        "whip_url":     "https://<your-machine-ip>:8443/movil_camera/whip"
    }
}
```

---

## Model Export Pipeline

The segmentation model must be exported from PyTorch to ONNX and compiled to a TensorRT engine before inference. Run these steps once from the `object-segmentation/` directory.

### Step 1 — Export to ONNX

```bash
cd object-segmentation/
python onnx_export.py
```

The `.onnx` file is written to `onnx_weights/`. Validate by comparing PyTorch and ONNX Runtime outputs — a mean absolute difference greater than `1e-3` indicates a problem. Do not proceed to TensorRT compilation until this is resolved.

### Step 2 — Compile to TensorRT

```bash
python export_tensorRT.py
```

The compiled engine is written to `tensorRT_weights/`. If the script fails, verify `libnvinfer` is on your library path:

```bash
echo $LD_LIBRARY_PATH
ldconfig -p | grep libnvinfer
```

---

## Running the System

Start each component in order.

### 1. MediaMTX (media relay)

Edit `rtsp-server/mediamtx.yml`:

```yaml
# Add stream paths
paths:
  movil_camera:
    rtspRangeType: clock
  movil_camera_2:
    rtspRangeType: clock

# Set your public IP
rtspAddress: <YOUR_PUBLIC_IP>:8554
webrtcAdditionalHosts: [<YOUR_PUBLIC_IP>]
webrtcICEServers2:
  - url: stun:stun.l.google.com:19302
  - url: stun:stun1.l.google.com:19302

# Reduce handshake latency
webrtcSTUNGatherTimeout: 3s
webrtcHandshakeTimeout: 3s
webrtcTrackGatherTimeout: 500ms

# Enable control API
api: true
```

Start the server:

```bash
cd rtsp-server/
./mediamtx mediamtx.yml
```

Verify the stream endpoint is reachable at `http://<host>:8889/movil_camera/`.

### 2. ML inference services

```bash
cd object-segmentation/

# Object segmentation (SAM-3 via TensorRT)
uvicorn sam_api:app --host <host> --port 8000

# Vision-language description (Gemma-4) — in a separate terminal
uvicorn gemma_api:app --host <host> --port 8001 
```

### 3. FastAPI proxy backend

```bash
python backend/proxy.py
```

### 4. Frontend HTTPS server

The frontend is served directly by the proxy. Open `https://<host>:<port>` on the mobile device.

---

## Usage

1. Open the frontend URL on the mobile device and grant camera permission.
2. The live camera feed begins streaming to MediaMTX; the processed overlay is rendered on-screen.
3. Touch and drag on the video canvas to draw a bounding box around an object.
4. Release to trigger SAM-3 segmentation. The mask is tracked across subsequent frames.
5. Tap the **+** icon at the corner of the mask to open the description panel — a cropped view of the object and a Gemma-4-generated description.
6. Tap **×** to close the panel.

---

## WebSocket Protocol

### Client → Server

| Event | Payload |
|---|---|
| `touch_start` | `{ x: float, y: float }` |
| `touch_move` | `{ x: float, y: float }` |
| `touch_end` | `{ x: float, y: float }` |
| `stream_start` | `{}` |
| `segment_object` | `{ bbox: { x1, y1, x2, y2 } }` |
| `request_description` | `{ frame_id: string }` |

All coordinates are normalized floats in `[0.0, 1.0]` relative to the frame dimensions.

### Server → Client

| Event | Payload |
|---|---|
| `overlay_stream` | Processed video stream |
| `tracking_update` | `{ mask_bbox: { x1, y1, x2, y2 } }` |
| `description_ready` | `{ text: string, segment_image_url: string }` |

---

## Project Structure

```
VISOR/
├── backend/
│   └── proxy.py                  # FastAPI reverse proxy — WebSocket + HTTP routing
├── frontend/
│   └── index.html                # Mobile web client (WebRTC, touch events, stream display)
├── object-segmentation/
│   ├── onnx_export.py            # Export SAM-3 from PyTorch to ONNX
│   ├── export_tensorRT.py        # Compile ONNX model to TensorRT engine
│   ├── demo_pytorch_webcam.py    # PyTorch inference (development/validation)
│   ├── demo_trt_webcam.py        # TensorRT inference (production)
│   ├── sam_api.py                # FastAPI service — segmentation + streaming
│   └── tensorRT_weights/         # Compiled TensorRT engine (populated after export)
├── vision_language_model/
│   └── test.py                   # Gemma-4 inference test
├── utils/
│   ├── cuda_handler.py           # TensorRT buffer allocation and inference helpers
│   ├── image_processing.py       # Mask overlay and bounding box drawing
│   ├── streaming.py              # OpenCV and VidGear streaming pipelines
│   ├── user_event.py             # Mouse/touch event handlers
│   ├── io.py                     # Config loading and logging setup
│   ├── wrappers.py               # Timing decorator
│   └── globals.py                # Shared application state
├── config.json.default           # Configuration template
├── DESIGN.MD                     # Architecture and design decisions
├── INSTALL.MD                    # Detailed setup guide
├── SCALABILITY.MD                # Concurrency, VRAM budgets, and horizontal scaling
├── SECURITY.MD                   # Security policy
├── OBSERVABILITY.MD              # Metrics, tracing, and alerting strategy
├── EFFICIENCY.MD                 # Performance optimization notes
├── BUSINESS_LINES.MD             # Target verticals and business model
├── RESEARCH_LINES.MD             # Research directions
└── LEGAL_RECOMMENDATION.MD       # Data protection and compliance framework
```

---

## Configuration Reference

### `config.json` (root)

| Key | Description |
|---|---|
| `certified` | Path to the TLS certificate file |
| `keyfile` | Path to the TLS private key file |
| `connection.host` | IP or hostname of the machine running the backend |
| `connection.port` | Port the backend listens on |
| `client.fastapi_http` | HTTPS base URL exposed to the browser client |
| `client.fastapi_ws` | WSS base URL exposed to the browser client |
| `client.whip_url` | WHIP endpoint for WebRTC camera ingestion |
| `sam_port` | Port of the SAM-3 inference service |
| `input_source` | RTSP URL of the incoming camera stream |
| `output_source` | RTMP URL for the processed output stream |

### `object-segmentation/config.json`

| Section | Description |
|---|---|
| `onnx_export` | Device, opset version, output path, sample image URL |
| `export_tensorRT` | Input ONNX path, output engine name |
| `demo_trt_webcam` | Engine path, prompt, output shapes, iteration limit |
| `sam_streaming_quality` | JPEG quality and target FPS for the streamed overlay |
| `sam_api_timeouts` | Connection and retry timeouts for the RTSP reader |

---

## Development Notes

**PyTorch fallback:** `demo_pytorch_webcam.py` runs inference directly with the Hugging Face model, without TensorRT. Use it to validate model behavior or on machines without TensorRT.

**TensorRT FP16:** The export script enables FP16 precision automatically when the GPU supports it. Optimization level is set to 5 for maximum engine performance at the cost of a longer build time (can take 30–60 minutes on first run).

**VRAM budget (RTX 5090, 32 GB):** SAM-3 consumes ~1.3 GB per session; Gemma-4 consumes ~5.0 GB per instance. Each concurrent user requires a dedicated SAM-3 instance. See `SCALABILITY.MD` for instance limits and scale-up behaviour.

**Session lifecycle:** Sessions are torn down on WebSocket disconnect or after 30 minutes, whichever comes first. All associated GPU memory is released at that point.

**Parallel inference:** Gemma-4 description generation runs independently of the SAM-3 segmentation loop to avoid blocking the video pipeline.

---

## Limitations (v1)

- Single object segmentation per session (no simultaneous multi-object tracking).
- No offline or on-device inference.
- No user authentication or session persistence.

---

## License

MIT — see [LICENSE](LICENSE).
