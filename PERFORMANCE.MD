# Performance

Under normal conditions the application has several bottlenecks that must be carefully studied to improve production costs, user experience, and quality of service.

---

## Connection Layer

The connection between a remote device and the server involves two points:

- **WebSocket** — data communication channel
- **WebRTC** — streaming connection

Both must be as fast and stable as possible. Currently, most streaming delay originates at the WebRTC ↔ remote device boundary. This needs to be improved, likely through hardware-specific encoding support (e.g. Qualcomm encoding, ARM-based devices, etc.).

As more devices are added to provide optimized service, connection complexity grows. Two approaches are worth considering:

- For **non-commercial users**: include a hardware recommendation guide in the product package.
- For **B2B**: supply the recommended hardware directly as an additional business offering.

---

## Server-Side Streaming Pipeline

The server faces the same hardware-specificity problem for handling streaming pipelines. A development path already exists using the **VidGear** interface to leverage FFMPEG's integrated hardware decoder/encoder (`h_cuda`, `h_nvenc`).

Example configuration:

```json
"input_vidgear": {
    "THREADED_QUEUE_MODE": false,
    "CAP_PROP_BUFFERSIZE": 1,
    "-avioflags": "direct",
    "-sync": "ext",
    "-fflags": "nobuffer",
    "-flags": "low_delay",
    "-analyzeduration": "0",
    "-probesize": "32",
    "-rtsp_transport": "udp",
    "-vcodec": null,
    "-enforce_cv_patch": true,
    "-ffprefixes": [
        "-vsync", "0",
        "-hwaccel", "cuda",
        "-hwaccel_output_format", "cuda"
    ],
    "-custom_resolution": "null",
    "-framerate": "null",
    "-vf": [
        "scale_cuda=640:480, hwdownload, format=nv12"
    ]
},
"output_vidgear": {
    "-vcodec": "h264_nvenc",
    "-f": "flv",
    "-preset": "p1",
    "-tune": "ll",
    "-rc": "cbr",
    "-g": "30",
    "-pix_fmt": "yuv420p",
    "-zerolatency": "1",
    "-delay": "0",
    "-b:v": "2500k",
    "-maxrate": "2500k",
    "-bufsize": "500k",
    "-rc-lookahead": "0",
    "-no-scenecut": "1",
    "-forced-idr": "1",
    "-bf": "0"
}
```

Beyond low-latency streaming, this approach frees up CPU and other resources for other tasks. **This optimization should also be applied to ML API streaming consumers.**

---

## ML API Performance

Pre-processing and post-processing of models is currently delegated to Hugging Face, which means performance costs in that layer cannot be directly modified. To address this, a new development effort is needed that clearly separates:

- **CPU-bound operations** — low arithmetic cost, latency-sensitive, benefits from higher clock speed.
- **Parallelizable operations** — operations with a naturally parallel structure (e.g. `CV.contours()`), to be rewritten in **C++/CUDA** and added as plugins in **TensorRT** compilations.

---

## Model Quantization

Model computation can be optimized from **FP16** to **INT8**, with expected accuracy loss below 2–3%. However, this requires a representative dataset reflecting real application usage patterns. This optimization is best deferred until real user data is available.

---

## TPU Migration

Switching to **TPU hardware** could significantly increase inference speed, but comes with substantial side conditions:

- Capital investment
- Pipeline changes
- Loss of streaming encoder/decoder support, which would need to be replaced

This should be treated as a **last-resort optimization**. See [`scalability.md`](./scalability.md) → *Hardware Selection* for side implications and hardware selection guidance.