from __future__ import annotations

from prometheus_client import Counter, Histogram

generate_requests_total = Counter(
    "wafer_generate_requests_total",
    "Total number of /generate API requests",
    labelnames=["defect_class", "sampler", "status"],
)

generate_latency_seconds = Histogram(
    "wafer_generate_latency_seconds",
    "Latency of /generate API requests in seconds",
    labelnames=["defect_class", "sampler"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

generate_images_total = Counter(
    "wafer_generate_images_total",
    "Total number of synthetic images generated",
    labelnames=["defect_class"],
)
