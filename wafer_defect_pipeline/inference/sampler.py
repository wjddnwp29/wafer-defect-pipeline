from __future__ import annotations

import time

import torch
from tqdm import tqdm

DEFAULT_NUM_CLASSES = 8


def _resolve_classes(
    classes: torch.Tensor | None,
    n_samples: int,
    device: torch.device,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> torch.Tensor:
    if classes is None:
        classes = torch.randint(0, num_classes, (n_samples,))
    return classes.to(device).long()


def generate_ddpm_with_nfe(
    ddpm,
    n_samples: int = 16,
    device: torch.device | None = None,
    classes: torch.Tensor | None = None,
    c: int = 1,
    h: int = 28,
    w: int = 28,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    if device is None:
        device = ddpm.device
    classes = _resolve_classes(classes, n_samples, device)

    x = torch.randn(n_samples, c, h, w, device=device)
    nfe = 0
    start = time.time()

    with torch.no_grad():
        ddpm.eval()
        for t in tqdm(range(ddpm.n_steps - 1, -1, -1), desc="DDPM", leave=False):
            time_tensor = torch.full((n_samples,), t, device=device).long()
            eta_theta = ddpm.backward(x, time_tensor, classes)
            nfe += 1

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            x = (1 / alpha_t.sqrt()) * (
                x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta
            )
            x = x.clamp(-1, 1)

            if t > 0:
                z = torch.randn_like(x)
                x = x + ddpm.betas[t].sqrt() * z
                x = x.clamp(-1, 1)

    return x, classes, nfe, time.time() - start


def generate_ddim_with_nfe(
    ddpm,
    n_samples: int = 16,
    device: torch.device | None = None,
    classes: torch.Tensor | None = None,
    ddim_steps: int = 50,
    eta: float = 0.0,
    c: int = 1,
    h: int = 28,
    w: int = 28,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    if device is None:
        device = ddpm.device
    classes = _resolve_classes(classes, n_samples, device)

    times = (
        torch.linspace(0, ddpm.n_steps - 1, steps=ddim_steps + 1)
        .round()
        .long()
        .to(device)
    )
    times = list(reversed(times.tolist()))
    time_pairs = list(zip(times[:-1], times[1:], strict=True))

    x = torch.randn(n_samples, c, h, w, device=device)
    nfe = 0
    start = time.time()

    with torch.no_grad():
        ddpm.eval()
        for t, prev_t in tqdm(time_pairs, desc=f"DDIM-{ddim_steps}", leave=False):
            time_tensor = torch.full((n_samples,), t, device=device).long()
            eta_theta = ddpm.backward(x, time_tensor, classes)
            nfe += 1

            alpha_bar_t = ddpm.alpha_bars[t]
            alpha_bar_prev = (
                ddpm.alpha_bars[prev_t]
                if prev_t >= 0
                else torch.tensor(1.0, device=device)
            )

            sigma_t = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                * (1 - alpha_bar_t / alpha_bar_prev)
            )

            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * eta_theta) / torch.sqrt(
                alpha_bar_t
            )
            pred_x0 = pred_x0.clamp(-1, 1)

            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * eta_theta

            noise = sigma_t * torch.randn_like(x) if sigma_t > 0 else torch.zeros_like(x)
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + noise

    return x, classes, nfe, time.time() - start


def generate_cm_with_nfe(
    cm,
    n_samples: int = 16,
    device: torch.device | None = None,
    classes: torch.Tensor | None = None,
    num_steps: int = 1,
    c: int = 1,
    h: int = 28,
    w: int = 28,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    if device is None:
        device = cm.device
    classes = _resolve_classes(classes, n_samples, device)

    nfe = 0
    start = time.time()

    with torch.no_grad():
        cm.eval()
        x = torch.randn(n_samples, c, h, w, device=device)

        if num_steps == 1:
            t = torch.full((n_samples,), cm.n_steps - 1, device=device).long()
            x = cm.consistency_function(x, t, classes)
            nfe = 1
        else:
            timesteps = (
                torch.linspace(cm.n_steps - 1, 0, num_steps + 1).long().tolist()
            )
            for i in range(num_steps):
                t_cur = timesteps[i]
                t_next = timesteps[i + 1]
                t = torch.full((n_samples,), t_cur, device=device).long()

                x0_pred = cm.consistency_function(x, t, classes)
                nfe += 1

                if t_next > 0:
                    alpha_bar_next = cm.alpha_bars[t_next]
                    noise = torch.randn_like(x)
                    x = (
                        torch.sqrt(alpha_bar_next) * x0_pred
                        + torch.sqrt(1 - alpha_bar_next) * noise
                    )
                else:
                    x = x0_pred

        x = x.clamp(-1, 1)

    return x, classes, nfe, time.time() - start
