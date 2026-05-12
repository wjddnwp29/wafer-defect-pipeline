from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from wafer_defect_pipeline.models.cm import ConsistencyModel, update_ema

DEFAULT_CD_STEP_SIZES: tuple[int, ...] = (1, 10, 50, 100, 200, 500, 700)


def consistency_distillation_loss(
    teacher_ddpm,
    student_cm,
    target_cm,
    x0: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    step_sizes: tuple[int, ...] = DEFAULT_CD_STEP_SIZES,
    recon_weight: float = 2.0,
    teacher_weight: float = 1.0,
) -> torch.Tensor:
    batch_size = x0.shape[0]
    n_steps = teacher_ddpm.n_steps

    u = torch.rand(batch_size, device=device)
    t = (u**2 * (n_steps - 1)).long() + 1
    t = t.clamp(1, n_steps - 1)

    total_consistency_loss = torch.zeros((), device=device)

    for step_size in step_sizes:
        t_prev = (t - step_size).clamp(0, n_steps - 1)

        noise = torch.randn_like(x0)
        alpha_bar_t = teacher_ddpm.alpha_bars[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

        with torch.no_grad():
            teacher_ddpm.eval()
            eps_pred = teacher_ddpm.network(x_t, t, y)
            pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
            pred_x0 = pred_x0.clamp(-1, 1)

            alpha_bar_prev = teacher_ddpm.alpha_bars[t_prev].view(-1, 1, 1, 1)
            x_t_prev = (
                torch.sqrt(alpha_bar_prev) * pred_x0
                + torch.sqrt(1 - alpha_bar_prev) * eps_pred
            )

            target_cm.eval()
            target_out = target_cm.consistency_function(x_t_prev, t_prev, y)

        student_out = student_cm.consistency_function(x_t, t, y)
        total_consistency_loss = total_consistency_loss + F.smooth_l1_loss(student_out, target_out)

    noise = torch.randn_like(x0)
    alpha_bar_t = teacher_ddpm.alpha_bars[t].view(-1, 1, 1, 1)
    x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
    student_out = student_cm.consistency_function(x_t, t, y)
    recon_loss = F.smooth_l1_loss(student_out, x0)

    with torch.no_grad():
        teacher_ddpm.eval()
        eps_pred = teacher_ddpm.network(x_t, t, y)
        teacher_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
        teacher_x0 = teacher_x0.clamp(-1, 1)
    teacher_loss = F.smooth_l1_loss(student_out, teacher_x0)

    return (
        total_consistency_loss / len(step_sizes)
        + recon_weight * recon_loss
        + teacher_weight * teacher_loss
    )


def train_consistency_model(
    teacher_ddpm,
    dataloader: DataLoader,
    n_epochs: int,
    device: torch.device,
    lr: float = 1e-4,
    store_path: str | Path = "cm_model.pt",
    ema_decay_max: float = 0.9999,
    ema_decay_min: float = 0.99,
    verbose: bool = True,
):
    student_network = copy.deepcopy(teacher_ddpm.network).to(device)
    student_cm = ConsistencyModel(
        student_network, n_steps=teacher_ddpm.n_steps, device=device
    )

    target_network = copy.deepcopy(teacher_ddpm.network).to(device)
    target_cm = ConsistencyModel(
        target_network, n_steps=teacher_ddpm.n_steps, device=device
    )
    target_cm.requires_grad_(False)
    target_cm.eval()

    optimizer = Adam(student_cm.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_loss = float("inf")
    loss_history: list[float] = []
    total_steps = n_epochs * len(dataloader)
    current_step = 0

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for batch in dataloader:
            x0 = batch[0].to(device)
            y = batch[1].to(device)

            loss = consistency_distillation_loss(
                teacher_ddpm, student_cm, target_cm, x0, y, device
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_cm.parameters(), max_norm=1.0)
            optimizer.step()

            ema_decay = ema_decay_max - (ema_decay_max - ema_decay_min) * (
                1 - current_step / total_steps
            )
            update_ema(target_cm.network, student_cm.network, decay=ema_decay)

            current_step += 1
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)

        log = f"Epoch {epoch + 1}/{n_epochs}: {avg_loss:.5f}"
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(student_cm.state_dict(), Path(store_path))
            log += " --> Best!"
        if verbose:
            print(log)

    student_cm.load_state_dict(torch.load(Path(store_path), map_location=device))
    return student_cm, loss_history
