"""Smoke tests that lock in the public surface of the package skeleton.

These pass as soon as imports resolve and types are constructable. As each
HW08 cell gets ported, replace the `pytest.skip` markers with real behavior
assertions.
"""

from __future__ import annotations


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


def test_unet_forward_smoke():
    import torch

    from wafer_defect_pipeline.models import MyUNet

    n_steps, num_classes, batch = 50, 8, 2
    net = MyUNet(n_steps=n_steps, time_emb_dim=100, num_classes=num_classes)
    x = torch.randn(batch, 1, 28, 28)
    t = torch.randint(0, n_steps, (batch,))
    y = torch.randint(0, num_classes, (batch,))
    out = net(x, t, y)
    assert out.shape == (batch, 1, 28, 28)


def test_ddpm_forward_smoke():
    import torch

    from wafer_defect_pipeline.models import MyDDPM, MyUNet

    n_steps, num_classes, batch = 50, 8, 2
    net = MyUNet(n_steps=n_steps, num_classes=num_classes)
    ddpm = MyDDPM(net, n_steps=n_steps, image_chw=(1, 28, 28))
    x0 = torch.randn(batch, 1, 28, 28)
    t = torch.randint(0, n_steps, (batch,))
    noisy = ddpm(x0, t)
    assert noisy.shape == x0.shape

    y = torch.randint(0, num_classes, (batch,))
    eps_pred = ddpm.backward(noisy, t, y)
    assert eps_pred.shape == x0.shape


def test_cm_forward_smoke():
    import torch

    from wafer_defect_pipeline.models import ConsistencyModel, MyUNet, update_ema

    n_steps, num_classes, batch = 50, 8, 2
    net = MyUNet(n_steps=n_steps, num_classes=num_classes)
    cm = ConsistencyModel(net, n_steps=n_steps)
    x = torch.randn(batch, 1, 28, 28)
    t = torch.randint(0, n_steps, (batch,))
    y = torch.randint(0, num_classes, (batch,))

    x0_pred = cm(x, t, y)
    assert x0_pred.shape == x.shape
    assert x0_pred.min() >= -1.0 and x0_pred.max() <= 1.0

    target = ConsistencyModel(MyUNet(n_steps=n_steps, num_classes=num_classes), n_steps=n_steps)
    update_ema(target, cm, decay=0.5)
