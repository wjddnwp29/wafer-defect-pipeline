"""Smoke tests that lock in the public surface of the package skeleton.

These pass as soon as imports resolve and types are constructable. As each
HW08 cell gets ported, replace the `pytest.skip` markers with real behavior
assertions.
"""

from __future__ import annotations

import pytest


def test_package_imports():
    import wafer_defect_pipeline  # noqa: F401
    from wafer_defect_pipeline import data, eval, inference, models, train, utils  # noqa: F401


def test_public_symbols_exposed():
    from wafer_defect_pipeline.data import ConditionalWaferDataset
    from wafer_defect_pipeline.eval import InceptionFeatureExtractor, calculate_fid
    from wafer_defect_pipeline.inference import (
        generate_cm_with_nfe,
        generate_ddim_with_nfe,
        generate_ddpm_with_nfe,
    )
    from wafer_defect_pipeline.models import ConsistencyModel, MyDDPM, MyUNet
    from wafer_defect_pipeline.train import train_consistency_model, training_loop
    from wafer_defect_pipeline.utils import seed_everything

    for obj in [
        ConditionalWaferDataset,
        InceptionFeatureExtractor,
        calculate_fid,
        generate_cm_with_nfe,
        generate_ddim_with_nfe,
        generate_ddpm_with_nfe,
        ConsistencyModel,
        MyDDPM,
        MyUNet,
        train_consistency_model,
        training_loop,
        seed_everything,
    ]:
        assert obj is not None


def test_seed_everything_runs():
    from wafer_defect_pipeline.utils import seed_everything

    seed_everything(0)


@pytest.mark.skip(reason="Pending Phase 1 port from HW08 cell #19/#22")
def test_ddpm_forward_smoke():
    """Replace once MyDDPM.forward is ported: 1-step shape preservation."""
