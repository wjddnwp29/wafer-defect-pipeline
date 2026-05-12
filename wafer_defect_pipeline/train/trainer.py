"""DDPM training loop.

Ported from HW08_20231049.ipynb (training_loop). Matplotlib plotting and CSV
dumping from the notebook are intentionally dropped; callers get loss_history
back and can persist whatever they need.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def training_loop(
    ddpm,
    loader: DataLoader,
    n_epochs: int,
    optim: torch.optim.Optimizer,
    device: torch.device,
    store_path: str | Path = "ddpm_model.pt",
    verbose: bool = True,
) -> list[float]:
    """Train a DDPM for n_epochs with eps-prediction MSE, saving the best checkpoint."""
    mse = nn.MSELoss()
    n_steps = ddpm.n_steps
    best_loss = float("inf")
    loss_history: list[float] = []

    epoch_iter = (
        tqdm(range(n_epochs), desc="Training", colour="#00ff00")
        if verbose
        else range(n_epochs)
    )

    for epoch in epoch_iter:
        epoch_loss = 0.0
        step_iter = (
            tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")
            if verbose
            else loader
        )

        for batch in step_iter:
            x0 = batch[0].to(device)
            y = batch[1].to(device)
            n = len(x0)

            eta = torch.randn_like(x0)
            t = torch.randint(0, n_steps, (n,), device=device)

            noisy = ddpm(x0, t, eta)
            eta_theta = ddpm.backward(noisy, t, y)

            loss = mse(eta_theta, eta)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * n / len(loader.dataset)

        loss_history.append(epoch_loss)

        log = f"Loss at epoch {epoch + 1}: {epoch_loss:.5f}"
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), Path(store_path))
            log += " --> Best model ever (stored)"
        if verbose:
            print(log)

    return loss_history
