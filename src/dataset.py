import numpy as np

def normalized_coordinates(height: int, width: int, frames: int) -> np.ndarray:
    """Produce x, y, t coordinates in [-1, 1] for a video volume."""
    y, x, t = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width), np.linspace(-1, 1, frames), indexing='ij')
    return np.stack([x, y, t], axis=-1).reshape(-1, 3)

def fourier_features(coords: np.ndarray, bands: int = 6) -> np.ndarray:
    freqs = 2.0 ** np.arange(bands)
    angles = np.pi * coords[..., None] * freqs
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1).reshape(len(coords), -1)
