# VISOR

Real-time object segmentation and natural language description system. Streams live video from a mobile browser to a GPU-accelerated backend, runs SAM-3 segmentation and PaliGemma description inference via TensorRT, and returns annotated overlays to the client over WebSocket.

---

## Overview

VISOR lets a user point their phone camera at an object, draw a bounding box around it, and receive a live segmentation mask with an AI-generated description — all in a browser, with no app installation required.

```
Mobile Browser ──WebRTC/WebSocket──▶ MediaMTX ──▶ Python Backend (GPU)
     │                                                      │
     └◀──────────── Processed overlay (WebSocket) ──────────┘
```

### Stack

| Layer | Technology |
|---|---|
| Frontend | Mobile web browser (WebRTC, Canvas) |
| Media relay | MediaMTX |
| Backend | Python 3.12 + FastAPI |
| Segmentation | SAM-3 (Meta) via TensorRT |
| Description | PaliGemma / Gemma-4 (Google) via Transformers |
| Transport | WebSocket + RTSP |
| Inference runtime | TensorRT (FP16), CUDA 12.9 |

---

## Requirements

- Python 3.12.11
- NVIDIA GPU with TensorRT support (tested on RTX 5090)
- CUDA 12.9 + cuDNN
- Linux (required for TensorRT)
- MediaMTX binary (see `rtsp-server/versions.txt` for the required version)
- FFmpeg binary (same `versions.txt`)
- Hugging Face account with access granted for:
  - [`facebook/sam3`](https://huggingface.co/facebook/sam3)
  - [`google/paligemma`](https://huggingface.co/google/paligemma) (or the Gemma-4 variant used in `vision_language_model/test.py`)

---

## Installation

### 1. Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Hugging Face authentication

Both models require explicit access approval on the Hugging Face model pages before downloading.

```bash
huggingface-cli login
```

See the [Hugging Face quickstart](https://huggingface.co/docs/huggingface_hub/quick-start) for details.

### 3. CUDA & cuDNN

Follow the [NVIDIA CUDA installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).

Verify your setup:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### 4. TLS certificate (required for camera access)

Modern browsers require HTTPS to access the device camera. Generate a self-signed certificate for development:

```bash
openssl x509 -in cert.csr -out cert.pem -req -signkey key.pem -days 1001
```

> **Note:** Self-signed certificates are for development only. Use a CA-issued certificate in production.

### 5. Configuration

Copy and fill in the root config:

```bash
cp config.json.default config.json
```

Edit `config.json`:

```json
{
    "certified": {
        "path": "server.pem"
    },
    "conection": {
        "host": "<your-machine-ip>",
        "port": 8443
    }
}
```

---

## Model Export Pipeline

The segmentation model must be exported from PyTorch to ONNX and then compiled to a TensorRT engine before inference. Run these steps once, from the `object-segmentation/` directory.

### Step 1 — Export to ONNX

```bash
cd object-segmentation/
python onnx_export.py
```

The exported `.onnx` file will be written to `onnx_weights/`. Inspect it visually with [Netron](https://github.com/lutzroeder/netron) if needed.

Validate the export by comparing PyTorch and ONNX Runtime outputs numerically. A mean absolute difference greater than `1e-3` indicates a problem — do not proceed to TensorRT compilation until this is resolved.

### Step 2 — Compile to TensorRT

```bash
python export_tensorRT.py
```

The compiled engine is written to `tensorRT_weights/`. If the script fails, verify that `libnvinfer` is on your library path:

```bash
echo $LD_LIBRARY_PATH        # should include the path to libnvinfer
ldconfig -p | grep libnvinfer
```

---

## Running the System

Start each component in order.

### 1. MediaMTX (media relay)

Edit `rtsp-server/mediamtx.yml`:

- Add the stream path entries at the end of the paths section:
  ```yaml
  movil_camera:
  server_overlay:
  ```
- Set the listening address to your public IP:
  ```yaml
  rtspAddress: <YOUR_PUBLIC_IP>:8554
  ```

Start the server:

```bash
cd rtsp-server/
./mediamtx mediamtx.yml
```

Verify the stream endpoint is reachable at `http://<host>:8889/movil_camera/`.

### 2. ML inference services

```bash
# Object segmentation (SAM-3 via TensorRT)
python sam_api.py

# Vision-language description (PaliGemma)
python gemma_api.py
```

### 3. FastAPI backend

```bash
python app.py
```

### 4. Frontend HTTPS server

```bash
cd frontend/
python python_server_https.py
```

The frontend is served at `https://<host>:<port>`. Open this URL on the mobile device.

---

## Usage

1. Open the frontend URL on the mobile device and grant camera permission.
2. The live camera feed begins streaming to MediaMTX; the processed overlay is rendered on-screen.
3. Touch and drag on the video canvas to draw a bounding box around an object.
4. Tap the bounding box to trigger SAM-3 segmentation. The mask is tracked across subsequent frames.
5. Tap the **+** icon at the corner of the mask to open the description panel, which shows a cropped view of the object and a PaliGemma-generated description.
6. Tap **×** to close the panel.

---

## WebSocket Protocol

### Client → Server

| Event | Payload |
|---|---|
| `stream_start` | `{}` |
| `draw_bbox` | `{ x1, y1, x2, y2 }` — normalized floats [0.0, 1.0] |
| `segment_object` | `{ bbox: { x1, y1, x2, y2 } }` |
| `request_description` | `{ frame_id: string }` |

All coordinates are normalized to `[0.0, 1.0]` relative to the frame dimensions.

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
├── app.py                          # FastAPI backend — WebSocket endpoints and processing loop
├── config.json.default             # Template for root configuration
├── frontend/
│   └── python_server_https.py      # HTTPS server for the mobile web frontend
├── object-segmentation/
│   ├── onnx_export.py              # Export SAM-3 from PyTorch to ONNX
│   ├── export_tensorRT.py          # Compile ONNX model to TensorRT engine
│   ├── infer_pytorch_webcam.py     # Inference with PyTorch (development/validation)
│   ├── infer_trt_webcam.py         # Inference with TensorRT engine (production)
│   ├── utils/common.py             # TensorRT buffer allocation and inference helpers
│   ├── logs/
│   ├── outputs/
│   └── tensorRT_weights/
└── vision_language_model/
    └── test.py                     # PaliGemma/Gemma-4 inference test
```

---

## Configuration Reference

### `config.json` (root)

Controls the frontend server.

| Key | Description |
|---|---|
| `certified.path` | Path to the TLS certificate file |
| `conection.host` | IP address or hostname of the machine running the frontend |
| `conection.port` | Port the frontend server listens on |

### `object-segmentation/config.json`

Controls model export and inference parameters. Key sections:

| Section | Description |
|---|---|
| `onnx_export` | Device, ONNX opset version, output path, sample image URL |
| `export_tensorRT` | Input ONNX path, output engine name |
| `infer_trt_webcam` | Engine path, prompt, output shapes, iteration limit |
| `metadata` | Tokenizer vocabulary and merge files |

---

## Development Notes

- **PyTorch fallback:** `infer_pytorch_webcam.py` runs inference directly with the Hugging Face model, without TensorRT. Use it to validate model behavior or when a GPU with TensorRT is not available.
- **TensorRT FP16:** The export script enables FP16 precision automatically when the GPU supports it (`builder.platform_has_fast_fp16`). Optimization level is set to 5 for maximum engine performance at the cost of a longer build time.
- **Frame/touch synchronization:** The backend matches touch events to video frames within a ±200 ms window using timestamps.
- **Parallel inference:** PaliGemma description generation runs independently of the SAM-3 segmentation and tracking loop to avoid blocking the video pipeline.

---


