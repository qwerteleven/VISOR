# Observability

This document describes the observability strategy for the system, covering where to find information about system status and potential incidents.

---

## Metrics Collection — Prometheus

Prometheus is the central metrics aggregation point for the system. Multiple components expose compatible endpoints that feed into a single Prometheus instance, enabling unified monitoring and future scalability.

### RTSP Server (MediaMTX)

MediaMTX includes a built-in Prometheus metrics interface. This must be explicitly enabled on the production system by editing `mediamtx.yml` and configuring the Prometheus scrape target accordingly.

### MinIO (Hot Storage)

MinIO also exposes a native Prometheus-compatible metrics endpoint. Like MediaMTX, this should be registered as a scrape target in the Prometheus configuration to centralize storage observability.

### Host Resource Monitoring (CPU, GPU, Disk, I/O, Network, RAM)

For monitoring physical and virtual host resources, any agent with compatibility with the QEMU/KVM virtualization stack is suitable. The recommended approach is to use an exporter that integrates with `libvirt` (exposed via `virt-manager`), so that both host and guest metrics can be scraped by Prometheus. This keeps all resource metrics in the same unified viewer alongside application-level metrics.

### FastAPI — Application Telemetry

FastAPI can be instrumented with **OpenTelemetry** to expose API-level telemetry: call rates, latency, error rates, and performance breakdowns. The OpenTelemetry collector can be configured to export metrics to Prometheus, keeping application telemetry in the same unified view.

---

## Distributed Tracing — Message Traceability

Messages and events exchanged between services must be fully traceable. Each message should carry enough context to identify:

- **Source** — which service or component originated it
- **Trigger** — the action or event that created it
- **User context** — any user or session implicated
- **Hop history** — the sequence of application points the message has passed through

This is critical for diagnosing non-systematic errors. Even when errors are correctly handled at a specific application point, or messages are intentionally discarded, tracing provides the only reliable way to identify the root cause. This is especially important in scenarios where individually handled errors accumulate and create systemic load problems — the errors may be "clean" from each service's perspective but still represent an unresolved issue at the system level.

---

## GPU Profiling — NVIDIA Visual Profiler

For production deployments running inference workloads, NVIDIA's Visual Profiler provides the necessary tooling to debug model engine execution. Key areas to monitor include:

- **Memory leaks** — tracking how GPU memory state evolves over time
- **Copy latency** — host-to-device and device-to-host transfer costs
- **Internal bottlenecks** — identifying pipeline stalls within the execution graph

---

## CPU / Code Profiling — Python cProfile

Python's built-in profiler can be used to measure resource usage and timing at the code level:

```bash
python -m cProfile -o system_metrics.out script.py
```

### Measurement Conditions

For meaningful profiling results, the application should be tested under two distinct conditions:

- **Idle** — baseline resource consumption with no active workload
- **Overloaded** — peak load to expose bottlenecks and worst-case behavior

### Important: Finite Execution

The profiler requires the code to terminate. Any `while True` infinite loops must be converted to a bounded number of iterations for profiling purposes. Take extra care when profiling API servers — the entry point and request handling need special consideration to produce valid, finite profiles.

---

## Pending Guidelines — To Be Developed

The following areas have been identified as necessary for a complete observability strategy. They are not yet implemented and require dedicated work to define and integrate.

---

### Centralized Log Aggregation

Log output across all services is handled by Python's built-in `logging` module using the following format:

```
%(levelname)s %(asctime)s %(processName)s %(message)s
```

What remains to be defined is a **centralized collection strategy** — where logs from all services are forwarded, indexed, and made queryable. Recommended candidates compatible with the existing Prometheus stack are **Loki** (with Grafana as the frontend) or an ELK/OpenSearch stack. The chosen solution should ingest structured log lines from all services, correlate them with trace IDs from the distributed tracing layer, and be available in the same unified dashboard as metrics.

---

### Dashboarding and Visualization — Grafana

> **Guideline:** To be developed.

Prometheus stores metrics but does not provide a usable visualization layer on its own. **Grafana** is the recommended frontend, as it natively integrates with Prometheus and Loki. Dashboards need to be designed to cover at minimum: host resources, RTSP stream health, MinIO storage, API performance, and GPU utilization. A single unified dashboard view is the target, in line with the scalability goals of the system.

---

### Alerting

> **Guideline:** To be defined by project leadership — requires directive with clear ownership and accountability.

The alerting strategy requires a formal decision from the responsible lead, covering: which metrics trigger alerts, severity levels, notification channels, and who is responsible for acknowledging and resolving each category of incident. **Alertmanager** (native to Prometheus) is the recommended tool for routing alerts. No thresholds or escalation paths should be configured without explicit directive, as incorrect alerting ownership creates ambiguity in incident response and accountability.

---

### Incident Runbooks

> **Guideline:** To be defined.

For each known failure mode (stream loss, storage saturation, API degradation, GPU memory exhaustion, etc.) a runbook must be written describing: how to confirm the incident, immediate mitigation steps, root cause investigation procedure, and resolution criteria. The distributed tracing layer (message traceability) is the primary diagnostic tool for non-systematic errors and should be the starting point in all runbooks.

---

### Baselines and Alerting Thresholds

> **Guideline:** Approximate approach defined — needs refinement over time.

Defining hard baselines is difficult at the current stage of the project given the early operational environment. The recommended interim approach is:

- **Soft baseline:** compute the rolling mean of each metric over the past month as a dynamic reference point.
- **Hard threshold:** flag any metric that exceeds **80% of its measured capacity** as a warning condition, regardless of historical mean. This covers systematic long-term degradation that a mean-based baseline would otherwise normalize over time.

Both baselines should be revisited and tightened as the system matures and accumulates sufficient operational history. Hard constraints should take priority over statistical baselines for metrics tied to capacity limits (disk, RAM, GPU memory), where gradual saturation represents a real systemic risk even if it appears "normal" historically.

Retention and storage policy for metrics and log data is defined in `scalability.md`.