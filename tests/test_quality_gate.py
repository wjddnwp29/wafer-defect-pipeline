from __future__ import annotations

import numpy as np
import pytest

from wafer_defect_pipeline.eval import (
    GateResult,
    calculate_fid,
    evaluate_fid_gate,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_gate_passes_when_fid_below_threshold(rng):
    real = rng.standard_normal((20, 16))
    fake = real + 0.01 * rng.standard_normal(real.shape)
    fid = calculate_fid(real, fake)

    result = evaluate_fid_gate(real, fake, threshold=fid + 1.0)
    assert isinstance(result, GateResult)
    assert result.passed is True
    assert result.fid == pytest.approx(fid)
    assert result.threshold == pytest.approx(fid + 1.0)


def test_gate_fails_when_fid_above_threshold(rng):
    real = rng.standard_normal((20, 16))
    far = rng.standard_normal((20, 16)) + 5.0
    fid = calculate_fid(real, far)

    result = evaluate_fid_gate(real, far, threshold=fid - 1.0)
    assert result.passed is False
    assert result.fid > result.threshold


def test_gate_passes_at_boundary(rng):
    real = rng.standard_normal((20, 16))
    fake = real + 0.01 * rng.standard_normal(real.shape)
    fid = calculate_fid(real, fake)

    result = evaluate_fid_gate(real, fake, threshold=fid)
    assert result.passed is True
