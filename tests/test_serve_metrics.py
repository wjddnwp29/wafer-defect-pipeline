from __future__ import annotations

from wafer_defect_pipeline.serve.metrics import (
    generate_images_total,
    generate_latency_seconds,
    generate_requests_total,
)


def _counter_value(counter, **labels) -> float:
    for metric_family in counter.collect():
        for sample in metric_family.samples:
            if sample.name.endswith("_total") and sample.labels == labels:
                return sample.value
    return 0.0


def test_generate_requests_total_increments():
    labels = {"defect_class": "Donut", "sampler": "ddim", "status": "success"}
    before = _counter_value(generate_requests_total, **labels)
    generate_requests_total.labels(**labels).inc()
    after = _counter_value(generate_requests_total, **labels)
    assert after == before + 1.0


def test_generate_images_total_increments_by_n():
    labels = {"defect_class": "Center"}
    before = _counter_value(generate_images_total, **labels)
    generate_images_total.labels(**labels).inc(8)
    after = _counter_value(generate_images_total, **labels)
    assert after == before + 8.0


def test_generate_latency_observe_increments_bucket():
    labels = {"defect_class": "Edge-Loc", "sampler": "ddim"}
    generate_latency_seconds.labels(**labels).observe(0.3)

    bucket_count = 0
    for metric_family in generate_latency_seconds.collect():
        for sample in metric_family.samples:
            if (
                sample.name.endswith("_bucket")
                and sample.labels.get("le") == "0.5"
                and sample.labels.get("defect_class") == "Edge-Loc"
            ):
                bucket_count = sample.value
    assert bucket_count >= 1.0


def test_metric_label_arity_enforced():
    import pytest

    with pytest.raises(ValueError):
        generate_requests_total.labels(defect_class="Donut")
