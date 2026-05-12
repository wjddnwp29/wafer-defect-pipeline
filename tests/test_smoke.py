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


def test_build_transform_smoke():
    import numpy as np
    from PIL import Image

    from wafer_defect_pipeline.data import build_transform

    tf = build_transform(img_size=28)
    img = Image.fromarray(np.zeros((40, 40), dtype=np.uint8), mode="L")
    out = tf(img)
    assert out.shape == (1, 28, 28)
    assert float(out.min()) >= -1.0 - 1e-6
    assert float(out.max()) <= 1.0 + 1e-6


def test_conditional_wafer_dataset_with_fake_pickle(tmp_path):
    import numpy as np
    import pandas as pd

    from wafer_defect_pipeline.data import ConditionalWaferDataset, build_transform

    fake_rows = []
    for label in ["Center", "Donut", "none"]:
        for _ in range(3):
            wmap = np.random.randint(0, 3, size=(32, 36), dtype=np.uint8)
            fake_rows.append({"waferMap": wmap, "failureType": np.array([[label]])})
    df = pd.DataFrame(fake_rows)
    pkl = tmp_path / "LSWMD.pkl"
    df.to_pickle(pkl)

    ds = ConditionalWaferDataset(pkl, transform=build_transform(28))
    assert len(ds) == 6  # 'none' dropped
    img, label = ds[0]
    assert img.shape == (1, 28, 28)
    assert label.dtype.is_floating_point is False
    assert set(ds.idx_to_class.values()) == {"Center", "Donut"}


def test_conditional_wafer_dataset_missing_file(tmp_path):
    import pytest

    from wafer_defect_pipeline.data import ConditionalWaferDataset

    with pytest.raises(FileNotFoundError):
        ConditionalWaferDataset(tmp_path / "does_not_exist.pkl")


def _tiny_loader(batch_size: int, n_batches: int, num_classes: int):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    n = batch_size * n_batches
    x = torch.randn(n, 1, 28, 28)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def test_training_loop_smoke(tmp_path):
    import torch

    from wafer_defect_pipeline.models import MyDDPM, MyUNet
    from wafer_defect_pipeline.train import training_loop

    device = torch.device("cpu")
    n_steps, num_classes = 20, 4
    net = MyUNet(n_steps=n_steps, num_classes=num_classes)
    ddpm = MyDDPM(net, n_steps=n_steps, device=device, image_chw=(1, 28, 28))
    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-3)

    ckpt = tmp_path / "ddpm.pt"
    history = training_loop(
        ddpm,
        _tiny_loader(batch_size=2, n_batches=2, num_classes=num_classes),
        n_epochs=2,
        optim=optim,
        device=device,
        store_path=ckpt,
        verbose=False,
    )
    assert len(history) == 2
    assert all(isinstance(loss, float) for loss in history)
    assert ckpt.exists()


def test_generate_ddpm_with_nfe_shape():
    import torch

    from wafer_defect_pipeline.inference import generate_ddpm_with_nfe
    from wafer_defect_pipeline.models import MyDDPM, MyUNet

    device = torch.device("cpu")
    n_steps, num_classes = 10, 4
    net = MyUNet(n_steps=n_steps, num_classes=num_classes)
    ddpm = MyDDPM(net, n_steps=n_steps, device=device, image_chw=(1, 28, 28))
    classes = torch.randint(0, num_classes, (3,))

    x, classes_out, nfe, _ = generate_ddpm_with_nfe(
        ddpm, n_samples=3, device=device, classes=classes
    )
    assert x.shape == (3, 1, 28, 28)
    assert classes_out.shape == (3,)
    assert nfe == n_steps


def test_generate_ddim_with_nfe_shape():
    import torch

    from wafer_defect_pipeline.inference import generate_ddim_with_nfe
    from wafer_defect_pipeline.models import MyDDPM, MyUNet

    device = torch.device("cpu")
    n_steps, num_classes = 20, 4
    net = MyUNet(n_steps=n_steps, num_classes=num_classes)
    ddpm = MyDDPM(net, n_steps=n_steps, device=device, image_chw=(1, 28, 28))
    classes = torch.randint(0, num_classes, (3,))

    x, _, nfe, _ = generate_ddim_with_nfe(
        ddpm, n_samples=3, device=device, ddim_steps=5, classes=classes
    )
    assert x.shape == (3, 1, 28, 28)
    assert nfe == 5


def test_generate_cm_with_nfe_shape():
    import torch

    from wafer_defect_pipeline.inference import generate_cm_with_nfe
    from wafer_defect_pipeline.models import ConsistencyModel, MyUNet

    device = torch.device("cpu")
    n_steps, num_classes = 20, 4
    net = MyUNet(n_steps=n_steps, num_classes=num_classes)
    cm = ConsistencyModel(net, n_steps=n_steps, device=device)
    classes = torch.randint(0, num_classes, (3,))

    x, _, nfe, _ = generate_cm_with_nfe(
        cm, n_samples=3, device=device, num_steps=1, classes=classes
    )
    assert x.shape == (3, 1, 28, 28)
    assert nfe == 1

    x, _, nfe, _ = generate_cm_with_nfe(
        cm, n_samples=3, device=device, num_steps=4, classes=classes
    )
    assert x.shape == (3, 1, 28, 28)
    assert nfe == 4


def test_show_images_returns_figure():
    import matplotlib

    matplotlib.use("Agg")
    import torch

    from wafer_defect_pipeline.utils import show_images

    imgs = torch.randn(4, 1, 28, 28)
    fig = show_images(imgs, title="t", labels=torch.tensor([0, 1, 2, 3]), show=False)
    assert fig is not None
    assert len(fig.axes) == 4


def test_calculate_fid_with_synthetic_features():
    import numpy as np

    from wafer_defect_pipeline.eval import calculate_fid

    rng = np.random.default_rng(0)
    real = rng.standard_normal((20, 16))
    fake = real + 0.01 * rng.standard_normal(real.shape)
    fid_close = calculate_fid(real, fake)

    far = rng.standard_normal((20, 16)) + 5.0
    fid_far = calculate_fid(real, far)

    assert isinstance(fid_close, float)
    assert fid_close < fid_far


def test_inception_feature_extractor_no_weights_shape():
    import torch

    from wafer_defect_pipeline.eval import InceptionFeatureExtractor, extract_features

    model = InceptionFeatureExtractor(weights=None).eval()
    imgs = torch.randn(2, 1, 28, 28)
    feats = extract_features(imgs, model, device=torch.device("cpu"), batch_size=2)
    assert feats.shape[0] == 2
    assert feats.ndim == 2
