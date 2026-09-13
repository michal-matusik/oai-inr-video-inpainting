"""Load official video frames, masks, coordinates, and validation rectangles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


VIDEO_LENGTH = 79


def load_observed_frames(root: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(root)
    paths = sorted((root / "images").glob("*.jpg"))
    times = np.asarray([int(path.stem) for path in paths], dtype=np.int64)
    images = np.stack([np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0 for path in paths])
    masks = np.stack([np.asarray(Image.open(root / "masks" / f"{path.stem}.png").convert("L"), dtype=np.float32) / 255.0 for path in paths])
    return times, images, masks


def load_rectangles(validation_root: str | Path) -> dict[str, dict[str, int]]:
    return json.loads((Path(validation_root) / "rectangles.json").read_text(encoding="utf-8"))


def normalized_coordinates(height: int, width: int, frames: int) -> np.ndarray:
    y, x, t = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width), np.linspace(0, 2, frames), indexing="ij")
    return np.stack([x, y, t], axis=-1).reshape(-1, 3)


def fourier_features(coords: np.ndarray, bands: int = 6) -> np.ndarray:
    frequencies = 2.0 ** np.arange(bands)
    angles = np.pi * coords[..., None] * frequencies
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1).reshape(len(coords), -1)
