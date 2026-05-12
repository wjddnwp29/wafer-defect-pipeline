from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch


def show_images(
    images,
    title: str = "",
    labels=None,
    class_names=None,
    show: bool = True,
) -> plt.Figure:
    if isinstance(images, tuple):
        images, labels = images

    if isinstance(images, torch.Tensor):
        images = images.cpu().detach()
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.cpu().detach()

    if isinstance(images, torch.Tensor):
        images = (images.clamp(-1, 1) + 1) / 2
        images = (images * 255).type(torch.uint8).numpy()

    n_images = len(images)
    rows = int(np.sqrt(n_images))
    cols = int(np.ceil(n_images / rows))

    fig = plt.figure(figsize=(12, 12))

    for idx in range(n_images):
        ax = fig.add_subplot(rows, cols, idx + 1)

        if images[idx].shape[0] == 1:
            ax.imshow(images[idx][0], cmap="gray")
        else:
            ax.imshow(images[idx].transpose(1, 2, 0))

        if labels is not None:
            label_idx = labels[idx].item() if hasattr(labels[idx], "item") else labels[idx]
            label_text = (
                class_names[label_idx] if class_names is not None else f"Class {label_idx}"
            )
            ax.set_title(label_text, fontsize=10, fontweight="bold")

        ax.axis("off")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def show_forward(ddpm, loader, device: torch.device) -> None:
    for batch in loader:
        imgs = batch[0]
        show_images(imgs, "Original images")

        n = len(imgs)
        for percent in [0.25, 0.5, 0.75, 1.0]:
            t = torch.full(
                (n,), int(percent * ddpm.n_steps) - 1, device=device, dtype=torch.long
            )
            noisy = ddpm(imgs.to(device), t)
            show_images(noisy, f"DDPM Noisy images {int(percent * 100)}%")
        break
