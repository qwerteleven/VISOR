# INSTALL.md — Setup & Deployment Guide

## Requirements

- **Python 3.12.11**
- NVIDIA GPU (tested on RTX 5090)
- CUDA 12.9 + cuDNN
- Linux (recommended for TensorRT support)

---

## 1. Python Dependencies

Install all Python dependencies from the requirements file:

```bash
pip install -r requirements.txt
```

---

## 2. Frontend

The frontend must be served over **HTTPS** because modern browsers require a secure context to access the device camera.

### Generate a Self-Signed Certificate

```bash
openssl x509 -in cert.csr -out cert.pem -req -signkey key.pem -days 1001
```

> ⚠️ This is a self-signed certificate for **development and testing only**.  
> For production, replace it with a certificate issued by a recognized Certificate Authority.  
> See: [https://github.com/openssl/openssl](https://github.com/openssl/openssl)

### Start the Frontend Server

```bash
cd frontend/
python python_server_https.py
```

---

## 3. RTSP Media Server (MediaMTX)

MediaMTX handles all streaming handshakes between the client and the server.

### Installation

1. Check `versions.txt` inside `rtsp-server/` for the required versions of **MediaMTX** and **FFmpeg**.
2. Download the corresponding binaries and place them inside the `rtsp-server/` folder.

### Configuration

Open `mediamtx.yml` and make the following changes:

**a) Add the stream path entries** at the end of the file:

```yaml
movil_camera:
server_overlay:
```

These define the incoming mobile camera stream and the outgoing processed overlay stream, respectively.

**b) Set the listening IP** to your public IP address to allow external connections:

```yaml
# Replace with your actual public IP
rtspAddress: <YOUR_PUBLIC_IP>:8554
```

### Start MediaMTX

```bash
cd rtsp-server/
./mediamtx mediamtx.yml
```

### Verify Incoming Stream

To confirm the mobile WebRTC stream is being received, open a browser and navigate to:

```
http://<host>:8889/movil_camera/
```

---

## 4. ML Backend

### 4.1 Model Access & Authentication

Both models require explicit access authorization before download.

**SAM-3 (Meta / Facebook)**
- Request access at: [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
- Authenticate with the Hugging Face CLI before the first download:
  ```bash
  huggingface-cli login
  ```
  See: [https://huggingface.co/docs/huggingface_hub/quick-start](https://huggingface.co/docs/huggingface_hub/quick-start)

**PaliGemma (Google)**
- Request access on Hugging Face under the Google/PaliGemma model page.
- Same Hugging Face authentication applies.

---

### 4.2 CUDA & cuDNN Setup

For best performance, the models should run compiled to TensorRT. This requires a working CUDA and cuDNN installation.

**Tested configuration:** RTX 5090 · CUDA 12.9

Follow the official NVIDIA installation guide:  
[https://docs.nvidia.com/cuda/cuda-installation-guide-linux/](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

**Verify GPU availability in Python:**

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

### 4.3 Export Model to ONNX

From the `object-segmentation/` directory, run:

```bash
python onnx_export.py
```

This generates an ONNX computation graph from the PyTorch model.

**Inspect the graph (optional):**  
Use Netron to visually inspect the exported `.onnx` file:  
[https://github.com/lutzroeder/netron](https://github.com/lutzroeder/netron)

**Validate the export:**  
Compare the numeric output of the original PyTorch model (before post-processing) with the output from the ONNX model using ONNX Runtime:  
[https://onnxruntime.ai/docs/get-started/with-python.html](https://onnxruntime.ai/docs/get-started/with-python.html)

> A small mean difference is acceptable. Any discrepancy **greater than `1e-3` is a red flag** — re-check the export before proceeding.

---

### 4.4 Compile to TensorRT

From the `object-segmentation/` directory, run:

```bash
python export_tensorrt.py
```

This script includes an auto-load plugin that handles TensorRT engine generation.

**Troubleshooting:**  
If the script fails, verify that `libnvinfer` is present and accessible via your environment variables:

```bash
echo $LD_LIBRARY_PATH   # should include the path to libnvinfer
ldconfig -p | grep libnvinfer
```

---

### 4.5 Start the ML APIs

Start each inference service independently:

```bash
# Object segmentation (SAM-3)
python sam_api.py

# Vision-language description (PaliGemma)
python gemma_api.py
```

---

## 5. Configuration

### `object-segmentation/config.json`

Controls all model-level settings:

- ONNX and TensorRT export parameters
- Segmentation overlay rendering (colors, opacity, stroke, etc.)
- File paths for model exports and artifacts

### `config.json` (project root)

Controls application-level settings:

- **`host`** — the IP address or hostname of the machine running the web frontend
- **`port`** — the port the frontend device will send messages to

> Make sure `host` and `port` match the network configuration of the device running the browser client.

---

## Summary Checklist

- [ ] Python 3.12.11 installed
- [ ] `pip install -r requirements.txt` completed
- [ ] Self-signed certificate generated (`cert.pem`, `key.pem`)
- [ ] Frontend HTTPS server running (`python_server_https.py`)
- [ ] MediaMTX binary downloaded (correct version per `versions.txt`)
- [ ] FFmpeg binary downloaded (correct version per `versions.txt`)
- [ ] `mediamtx.yml` updated with stream paths and public IP
- [ ] MediaMTX service running
- [ ] Hugging Face access granted for SAM-3 and PaliGemma
- [ ] `huggingface-cli login` completed
- [ ] CUDA 12.9 + cuDNN installed and verified
- [ ] ONNX export completed and validated (`< 1e-3` error)
- [ ] TensorRT engine compiled (`export_tensorrt.py`)
- [ ] `libnvinfer` accessible in environment path
- [ ] `sam_api.py` running
- [ ] `gemma_api.py` running
- [ ] `config.json` (root) updated with correct `host` and `port`
