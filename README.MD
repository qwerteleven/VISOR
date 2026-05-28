# SAM3-TensorRT

Fast inference pipeline for [Meta's SAM3](https://huggingface.co/facebook/sam3) (Segment Anything Model 3) using [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt), with support for ONNX export, webcam streaming, and PyTorch reference inference.

---

## Overview

SAM3 is an open-vocabulary instance segmentation model. This repo provides tooling to:

- Export SAM3 from HuggingFace to [ONNX](https://onnx.ai/) format
- Compile the ONNX model to a TensorRT engine for fast GPU inference
- Run inference on static images or live webcam feeds
- Visualize segmentation masks and bounding boxes

---

## Project Structure

```
SAM3-TensorRT/
├── export_tensorRT.py          # Compile ONNX model → TensorRT engine
├── infer_onnx.py               # Static image inference via ONNX Runtime
├── infer_pytorch_webcam.py     # Webcam inference via HuggingFace PyTorch
├── infer_trt.py                # Static image inference via TensorRT engine
├── infer_trt_webcam.py         # Webcam inference via TensorRT engine
├── onnx_export.py              # Export SAM3 from HuggingFace to ONNX
├── test_hugging_face_model_access.py  # Verify HuggingFace model access
├── config.json                 # Shared configuration for all scripts
└── utils/
    ├── common.py               # TensorRT buffer allocation & CUDA helpers
    ├── detect_postprocess.py   # Mask/box post-processing and visualization
    ├── image_preprocess.py     # Image resizing and normalization
    ├── pytorch_visualize.py    # PIL-based result visualization
    └── simplify_tokenizer.py   # Standalone CLIP BPE tokenizer
```

---

## Pipeline

```
HuggingFace (facebook/sam3)
        │
        ▼
  onnx_export.py          →  onnx_weights/<model>.onnx
        │
        ▼
  export_tensorRT.py      →  <model>.engine
        │
        ├──▶ infer_trt.py            (static image)
        └──▶ infer_trt_webcam.py     (webcam stream)
```

For reference or development without TensorRT, `infer_onnx.py` and `infer_pytorch_webcam.py` provide alternative inference paths.

---

## Dependencies

### Core Inference

| Library | Purpose |
|---|---|
| [TensorRT](https://developer.nvidia.com/tensorrt) | High-performance GPU inference engine |
| [ONNX Runtime](https://onnxruntime.ai/) | Cross-platform ONNX inference |
| [PyTorch](https://pytorch.org/) | Model loading, export, and reference inference |
| [CUDA Python (`cuda-python`)](https://nvidia.github.io/cuda-python/) | Low-level CUDA memory management in `utils/common.py` |

### Model & Tokenizer

| Library | Purpose |
|---|---|
| [Transformers (HuggingFace)](https://huggingface.co/docs/transformers) | `Sam3Model`, `Sam3Processor` |
| [huggingface_hub](https://huggingface.co/docs/huggingface_hub) | Model authentication and download |
| [regex](https://pypi.org/project/regex/) | Unicode-aware BPE tokenization in `SimpleCLIPBPETokenizer` |

The tokenizer is a simplified implementation of the [CLIP BPE tokenizer](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTokenizer), requiring a `vocab.json` and `merges.txt` from the SAM3 model files.

### Image Processing & Visualization

| Library | Purpose |
|---|---|
| [OpenCV (`cv2`)](https://opencv.org/) | Image I/O, resizing, webcam capture, drawing |
| [NumPy](https://numpy.org/) | Array operations throughout the pipeline |
| [Pillow (PIL)](https://pillow.readthedocs.io/) | RGBA compositing for mask overlays in PyTorch visualizer |

---

## Configuration

All scripts read from `config.json`. Each top-level key corresponds to a script:

```json
{
  "metadata": {
    "vocab_file": "models/vocab.json",
    "merges_file": "models/merges.txt"
  },
  "onnx_export": {
    "device": "cuda",
    "prompt": "person",
    "image_url": "...",
    "onnx_path": "sam3.onnx",
    "dynamo": false,
    "opset_version": 18
  },
  "export_tensorRT": {
    "model_path": "onnx_weights/sam3.onnx",
    "output_name": "sam3.engine"
  },
  "infer_trt": {
    "image_url": "demo/image.jpg",
    "engine_file_path": "sam3.engine",
    "prompt": "person",
    "outputs_shapes": [[1, 200, 256, 256], [1, 200, 4], [1, 200]],
    "score_thr": 0.4,
    "mask_thr": 0.5,
    "max_inst": 30,
    "boxes_normalized": true,
    "output_image": "output.jpg"
  },
  "infer_trt_webcam": { "..." : "..." },
  "infer_onnx": { "..." : "..." },
  "infer_pytorch_webcam": { "..." : "..." }
}
```

---

## Usage

### 1. Verify model access

```bash
python test_hugging_face_model_access.py
```

### 2. Export to ONNX

```bash
python onnx_export.py
```

### 3. Compile TensorRT engine

```bash
python export_tensorRT.py
```

FP16 is enabled automatically if the GPU supports it. Optimization level is set to 5 (longest build, fastest runtime).

### 4. Run inference

**TensorRT — static image:**
```bash
python infer_trt.py
```

**TensorRT — webcam:**
```bash
python infer_trt_webcam.py
```

**ONNX Runtime — static image:**
```bash
python infer_onnx.py
```
> ⚠️ ONNX inference has a known post-processing dimension mismatch and is a work in progress.

**PyTorch — webcam:**
```bash
python infer_pytorch_webcam.py
```

---

## Key Concepts

- **[Instance Segmentation](https://paperswithcode.com/task/instance-segmentation)** — detecting and masking individual object instances, as opposed to semantic segmentation which labels pixels by class only.
- **[ONNX Export](https://pytorch.org/docs/stable/onnx.html)** — converting a PyTorch model to the open ONNX interchange format, enabling deployment outside of PyTorch.
- **[TensorRT Engine Compilation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)** — optimizing an ONNX graph for a specific GPU, including kernel fusion, precision calibration (FP16/INT8), and layer tuning.
- **[CLIP BPE Tokenizer](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTokenizer)** — byte-pair encoding tokenizer used to encode open-vocabulary text prompts into token IDs and attention masks for the model.
- **[FP16 Inference](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#mixed-precision)** — running model weights and activations in 16-bit floating point for approximately 2× throughput on supported NVIDIA GPUs (Turing and later).

---

## Notes

- The model is wrapped in `Sam3ONNXWrapper` during export to produce clean `pixel_values / input_ids / attention_mask` inputs and `pred_masks / semantic_seg / pred_logits` outputs.
- `utils/common.py` is derived from [NVIDIA's TensorRT Python samples](https://github.com/NVIDIA/TensorRT/tree/main/samples/python) and handles pinned-memory buffer allocation and async CUDA memcpy.
- Webcam inference benchmarks print average frame time every 100 iterations.