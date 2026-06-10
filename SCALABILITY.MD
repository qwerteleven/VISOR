# Scalability Analysis

## ML Inference Pipeline — GPU/TPU Constraints

The most restrictive component is the connection between the user and the ML segmentation model. Due to hardware limitations, the model must be computed server-side to serve user devices. This means every connected user triggers a unique ML model execution on the GPU/TPU, with both streaming input and streaming output.

**Current VRAM consumption:**

| Model | Precision | VRAM |
|-------|-----------|------|
| SAM3 | FP16 | 1.3 GB |
| Gemma4 | FP16 | 5.0 GB |

The primary bottleneck is **SAM3**, which can only handle **one streaming session at a time**. Gemma4, by contrast, supports many-to-many request handling and runs as a fully parallel process — its inference is agnostic to SAM3 sequencing. The current design does not require extended context — a single one-shot inference is sufficient.

---

## Reference Hardware — RTX 5090

| Spec | Value |
|------|-------|
| VRAM | 32 GB |
| Compute (FP16) | 200 TFLOPs |
| Memory Bandwidth | 1.79 TB/s |

---

## Input/Output Memory Throughput

A single streaming frame at the current SAM3 input resolution:

```
1008 × 1008 × 3 channels × 4 bytes (float32) ≈ 12.2 MB per frame
```

At this size, neither memory bandwidth nor raw compute power is the limiting factor — the single-stream constraint of SAM3 is the actual ceiling.

---

## SAM3 & Gemma4 Concurrency — Triton Inference Server

The lifecycle of all model instances — initialization, load, and unload — is managed by a **Triton Inference Server**.

### SAM3

Each new user session spawns a **dedicated SAM3 instance** (1.3 GB VRAM each). There is no shared queue for SAM3 — every user gets their own instance. When a new instance cannot be created (VRAM exhausted on the current VM), the request is redirected to the **VM consumer network** to be handled by another VM. If no VM in the network can accommodate the request, it is discarded and the user receives an **"overload capacity"** error. The service responsible is also notified.

### Gemma4

Gemma4 operates as a shared many-to-many service. Overload is defined as a message with **more than 1 minute of delay between input and output**. When this threshold is crossed, Triton initializes an additional Gemma4 instance. The same escalation path applies if a new instance cannot be created: redirect to the VM network, then discard with error and notification if all capacity is exhausted.

### VRAM Ceiling — RTX 5090

```
32 GB total
 - 1.3 GB × N  (SAM3 instances)
 - 5.0 GB × M  (Gemma4 instances)
```

| SAM3 instances | Gemma4 instances | VRAM used | Remaining |
|---------------|-----------------|-----------|-----------|
| 1 | 1 | 6.3 GB | 25.7 GB |
| 4 | 4 | 25.2 GB | 6.8 GB |
| 6 | 4 | 27.8 GB | 4.2 GB |
| 16 | 2 | 30.8 GB | 1.2 GB |

Triton must enforce hard instance limits before VRAM exhaustion reaches the driver level. Proactive notifications to the service responsible must be triggered at **70%, 80%, and 90% VRAM utilization** to allow intervention before capacity is fully consumed.

---

## Session Lifecycle

Sessions are destroyed by two mechanisms:

- **Maximum computation time**: Sessions are forcibly terminated after **30 minutes** regardless of activity.
- **WebSocket disconnection**: When the user's WebSocket connection closes, the session and its associated SAM3 instance are torn down immediately.

On destruction, the HuggingFace preprocessing/postprocessing objects bound to that session are released, and Triton unloads the model instance if no longer needed.

---

## Preprocessing / Postprocessing Architecture

Part of the inference logic is delegated to the **HuggingFace preprocessing and postprocessing pipeline**, which must be instantiated as a **separate object per user session**. While the core SAM3 API can be shared across users, each user's preprocessing/postprocessing state must remain isolated.

A **per-session token** should be issued to differentiate computation fluxes between users and avoid state leakage between concurrent sessions.

---

## CPU & Network Bottlenecks

The CPU carries the majority of non-GPU services:

- RTSP server
- HTTPS web server
- PostgreSQL database
- Coordination of ML inference inputs/outputs

This stack makes the **CPU a likely bottleneck** at scale. The RTSP server's scalability has not been tested under realistic load due to resource constraints.

In addition, concurrent connections may saturate network infrastructure. Mitigation strategies to consider:

- Load balancing policy
- Redundant connection servers
- Connection pooling / rate limiting per client

---

## Data Storage Growth

Application output is stored in two layers:

- **PostgreSQL**: Stores metadata references (object IDs, session linkage, descriptions)
- **MinIO (object storage)**: Stores the actual binary data — segmentation masks and generated descriptions

**Estimated output per inference:**

| Output | Size |
|--------|------|
| ~30 segmentation masks at 640×640 | ~30 × 640 × 640 × 1 byte ≈ ~12.3 MB (binary mask) |
| Gemma4 text description | Negligible |

Storage growth is **unbounded and usage-driven** — no fixed ceiling can be defined without usage projections. Capacity planning should account for mask storage as the dominant factor, with text descriptions being anecdotal in size.

---

## Storage Retention Policy

To control unbounded growth, a two-phase retention policy is applied:

1. **Similarity deduplication**: Before deleting, a similarity metric is computed between masks. Redundant or near-duplicate segmentations are candidates for early removal.
2. **Rotary deletion by age**: Among the surviving masks, the oldest segmentations are progressively deleted on a rolling basis.

### Cold Storage Tiering

MinIO applies a **crontab-scheduled policy** to migrate old data to a **separate NAS** (cold HDD storage tier) before permanent removal from the hot storage pool. The NAS is an independent unit and constitutes its own failure domain.

If the NAS also reaches full capacity, the data slated for archival is **permanently deleted** and the service responsible is notified.

### Notification Thresholds

Both MinIO and NAS capacity must trigger proactive notifications **before** reaching hard limits:

| Threshold | Action |
|-----------|--------|
| 70% full | Notify service responsible |
| 80% full | Notify service responsible (escalation) |
| 90%+ full | Notify service responsible (critical) |
| 100% full (NAS) | Delete data, notify service responsible |

PostgreSQL metadata records follow the same lifecycle as their corresponding MinIO objects.

---

## Failure Modes & Recovery

The application is **not replicated**. Recovery behavior by failure type:

| Failure | Behavior |
|---------|----------|
| GPU OOM / SAM3 crash | Application state is reset to the initial step; the session is torn down and must be restarted by the user |
| MinIO unreachable | A warning is displayed to the user; the inference workflow **continues** without storage — results are not persisted until storage recovers |
| PostgreSQL unreachable | Undefined — should be treated similarly to MinIO with a graceful degradation path |
| WebSocket drop | Session is destroyed immediately (see Session Lifecycle) |

Because the system is unreplicated, any stateful failure requires a full restart from step one. There is no partial recovery or checkpoint mechanism at this time.

---

## Horizontal Scaling

The application is designed to run inside a **QEMU/KVM virtualization environment**. On-demand VMs are created dynamically, each with a fixed slice of GPU, RAM, and CPU resources assigned at provisioning time.

### Scaling Trigger — KVM Metrics

A new VM is provisioned when **any active VM hits 80% utilization** on any of the following metrics:

| Metric | Scale-up threshold |
|--------|--------------------|
| CPU usage | ≥ 80% |
| VRAM usage | ≥ 80% |
| Disk I/O / usage | ≥ 80% |

As with VRAM, proactive notifications to the service responsible are sent at **70%, 80%, and 90%** utilization for each metric, allowing intervention before the ceiling is hit.

If no physical resources remain to provision a new VM, the user receives an **"overload capacity"** error and the service responsible is notified.

### Caveats

- GPU partitioning in KVM environments (via vGPU or PCIe passthrough) adds significant operational complexity.
- VRAM slicing is not flexible at runtime — each VM's GPU allocation is fixed at creation.
- This approach provides **hardware-level isolation** between tenants but makes dynamic reallocation coarse-grained; spinning a VM is a slow, heavyweight operation compared to container-level scaling.
- On a single physical host, this architecture functions as a **resource governor** rather than true horizontal scaling. True horizontal scaling would require multiple physical hosts with a shared network fabric, which is outside the current hardware scope.