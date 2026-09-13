"""Torch model kept separate so coordinate preprocessing remains testable without Torch."""
import torch
from torch import nn

class SineLayer(nn.Module):
    def __init__(self, input_dim, output_dim, omega=30.0):
        super().__init__(); self.linear = nn.Linear(input_dim, output_dim); self.omega = omega
    def forward(self, x): return torch.sin(self.omega * self.linear(x))

class VideoINR(nn.Module):
    def __init__(self, feature_dim=36, width=256, depth=4):
        super().__init__()
        layers = [SineLayer(feature_dim, width)] + [SineLayer(width, width) for _ in range(depth - 1)]
        self.backbone = nn.Sequential(*layers); self.rgb = nn.Linear(width, 3); self.mask = nn.Linear(width, 1)
    def forward(self, features):
        latent = self.backbone(features)
        return torch.sigmoid(self.rgb(latent)), self.mask(latent)
