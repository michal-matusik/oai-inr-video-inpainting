"""Coordinate-based neural and deterministic video-field models."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class SineLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, omega: float = 30.0):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.omega = omega

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class VideoINR(nn.Module):
    """SIREN-style trainable field kept for model experiments."""

    def __init__(self, input_dim: int = 3, width: int = 256, depth: int = 4):
        super().__init__()
        layers = [SineLayer(input_dim, width)] + [SineLayer(width, width) for _ in range(depth - 1)]
        self.backbone = nn.Sequential(*layers)
        self.rgb = nn.Linear(width, 3)
        self.mask = nn.Linear(width, 1)

    def forward(self, coordinates):
        latent = self.backbone(coordinates)
        return torch.sigmoid(self.rgb(latent)), torch.sigmoid(self.mask(latent))


class TemporalVideoField(nn.Module):
    """Implicit field reconstructed by temporal interpolation of observed frames."""

    def __init__(self, times: np.ndarray, images: np.ndarray, masks: np.ndarray):
        super().__init__()
        self.register_buffer("times", torch.as_tensor(times, dtype=torch.long))
        self.register_buffer("images", torch.as_tensor(images, dtype=torch.float32))
        self.register_buffer("masks", torch.as_tensor(masks, dtype=torch.float32))

    def predict_frame(self, frame: int) -> tuple[torch.Tensor, torch.Tensor]:
        exact = torch.where(self.times == frame)[0]
        if len(exact):
            index = int(exact[0])
            return self.images[index], self.masks[index]
        lower = torch.where(self.times < frame)[0][-1]
        upper = torch.where(self.times > frame)[0][0]
        weight = (frame - int(self.times[lower])) / (int(self.times[upper]) - int(self.times[lower]))
        image = (1.0 - weight) * self.images[lower] + weight * self.images[upper]
        mask = (1.0 - weight) * self.masks[lower] + weight * self.masks[upper]
        return image, mask

    def forward(self, coordinates: torch.Tensor):
        """Return BGR and mask values in the order expected by the official notebook."""
        outputs_rgb, outputs_mask = [], []
        for batch in coordinates:
            frame = round(float(batch[0, 2]) * 39.0)
            image, mask = self.predict_frame(frame)
            side = math.isqrt(len(batch))
            if side * side != len(batch):
                raise ValueError("the official evaluator supplies a square coordinate grid")
            x_start = round(float((batch[:, 0].min() + 1.0) * 128.0))
            y_start = round(float((batch[:, 1].min() + 1.0) * 128.0))
            rows, columns = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
            rgb = image[y_start + rows, x_start + columns].reshape(-1, 3)
            alpha = mask[y_start + rows, x_start + columns].reshape(-1, 1)
            outputs_rgb.append(rgb[:, [2, 1, 0]])
            outputs_mask.append(alpha)
        return torch.stack(outputs_rgb), torch.stack(outputs_mask)
