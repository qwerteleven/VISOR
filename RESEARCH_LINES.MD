# Research Lines


## 1. Box Segmentation Tracking 

The application uses a user-defined bounding box to guide SAM3 in segmenting a target object.
However, in remote device environments (phone, tablet, etc.), device movement causes the box to
lose alignment with the intended target.

Lines 1 and 2 address the same root problem — box drift on moving devices — through two
complementary approaches: model-side tracking (line 1) and sensor-driven geometric correction
(line 2). Both should be explored in parallel and potentially combined.

**Research directions:**

- Evaluate the SAM3 variant that includes a native tracker
- Develop a proprietary tracking pipeline using segmentation coincidence metrics such as IoU
- Analyze the correlation between time and false positives
- Apply a **Kalman filter** for signal stabilization

---

## 2. Device-Aware Box Transformation 

Complementary to line 1, this approach maintains correct box tracking despite device movement
by exploiting onboard sensor data directly, rather than relying on the segmentation model.

- Use **GPS** position data and **gyroscope** readings to compute translations and transformations
  of the detection box
- Explore photogrammetry techniques for geometric correction
- Reference: [cameratransform](https://github.com/rgerum/cameratransform)

---

## 3. VLM Evaluation — Gemma4 and Alternatives

With the accelerating growth of Vision-Language Models, Gemma4 must be continuously benchmarked
against alternative options for description generation.

- Reference: [arxiv.org/abs/2512.10942](https://arxiv.org/abs/2512.10942)

**Two distinct risks to track separately:**

>  **Risk A — Model obsolescence:** The VLM landscape evolves fast enough that Gemma4 may be
> overtaken by superior alternatives. Benchmarking must be an ongoing activity, not a one-time
> evaluation.

>  **Risk B — Upper layer lock-in:** If the layer built on top of Gemma4 grows too large or
> too tightly coupled, migrating to a better model becomes prohibitively costly. Architectural
> decoupling from the underlying VLM must be a first-class design constraint.

---

## 4. RAG Layer for Domain-Specific Segmentation 

A RAG layer in conjunction with the VLM to improve responses for image-segmentation requests
directed to Gemma4. The architecture centers on an **object classifier** that determines the
category of the requested description and routes it to the appropriate domain-specific RAG
support module, providing Gemma4 with targeted contextual grounding.

### Object Detector Scope

The classifier requires an object detector, and the choice of detector must be scoped to the
deployment context:

- **Server-side (full resources):** A SOTA model such as **DINO-DETR** can be used for maximum
  accuracy.
- **Resource-constrained fallback:** If DINO-DETR generates unacceptable system load,
  it can be replaced by **YOLOv12** as a lighter alternative.

Regardless of the detector chosen, an **intermediate embedding layer** is required to correctly
translate between the model's internal representation space and the vector space used by the RAG
index. This translation layer is non-trivial and must be explicitly designed and validated —
a direct mismatch between detector embeddings and RAG vectors will silently degrade retrieval
quality.

**Issues identified:**

- Hard coupling between RAG embedding alignment and Gemma4
- The intermediate embedding translator adds an additional architectural dependency
- The speed of VLM advancement risks keeping this solution permanently out of production

> **Status: Open and conditional.** This line must be continuously evaluated against the pace
> of VLM development. It should be closed only if its progress falls out of timing alignment with
> the broader research — i.e. if VLM evolution makes it obsolete before it reaches production
> readiness.

---

## 5. SAM3 Follow-Up and Efficiency Optimization 

Continuous monitoring of SAM3 development and its alternatives is essential, as it constitutes
the core of the application.

**Focus areas:**

- Track improvements and new releases of SAM3 and competing segmentation models
- Investigate concrete optimizations targeting the internal complexity of LLM/VLM architectures
- Reference on complexity reduction (n⁴ attention): [arxiv.org/abs/2504.19874](https://arxiv.org/abs/2504.19874)

---

## 6. Streaming Inpainting of Segmentations 

A side research line exploring the possibility of streaming stored inpainting segmentations to
enable real-time world modification before performing a constrained action.

**Potential use cases:**

- Previewing changes to public space decoration
- Visualizing environment redesigns (offices, homes, retail spaces)
- Planning industrial layout changes
- Copy-paste machinery placement with real-time spatial preview

- Reference: [arxiv.org/abs/2603.05507](https://arxiv.org/abs/2603.05507)