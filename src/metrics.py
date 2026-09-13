"""Official validation metrics and score conversion."""

import numpy as np


def psnr(target: np.ndarray, prediction: np.ndarray) -> float:
    mse = float(np.mean((target - prediction) ** 2))
    return float("inf") if mse == 0 else -10.0 * np.log10(mse)


def binary_accuracy(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((prediction > 0.5) == (target > 0.5)))


def task_score(mean_psnr: float, mean_accuracy: float) -> int:
    psnr_points = 0.0 if mean_psnr < 15.5 else min(10.0, (mean_psnr - 15.5) * 5.0 / 4.0)
    accuracy_points = 0.0 if mean_accuracy < 0.83 else min(1.0, (mean_accuracy - 0.83) * 20.0 / 3.0)
    return round(7.0 * psnr_points + 30.0 * accuracy_points)
