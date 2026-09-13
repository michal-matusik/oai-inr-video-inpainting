"""Evaluate on the public held-out frames without fitting to their contents."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import load_rectangles
from .metrics import binary_accuracy, psnr, task_score
from .train import train


def evaluate(train_root: str, validation_root: str):
    model = train(train_root)
    validation = Path(validation_root)
    psnrs, accuracies = [], []
    for name, rectangle in load_rectangles(validation).items():
        frame = int(Path(name).stem)
        prediction, predicted_mask = model.predict_frame(frame)
        target = np.asarray(Image.open(validation / "images" / name).convert("RGB"), dtype=np.float32) / 255.0
        mask_name = name.replace(".jpg", ".png")
        target_mask = np.asarray(Image.open(validation / "masks" / mask_name).convert("L"), dtype=np.float32) / 255.0
        x1, y1, x2, y2 = (rectangle[key] for key in ("x1", "y1", "x2", "y2"))
        psnrs.append(psnr(target[y1:y2, x1:x2], prediction[y1:y2, x1:x2].numpy()))
        accuracies.append(binary_accuracy(target_mask[y1:y2, x1:x2], predicted_mask[y1:y2, x1:x2].numpy()))
    mean_psnr, mean_accuracy = float(np.mean(psnrs)), float(np.mean(accuracies))
    return {"psnr_db": mean_psnr, "mask_accuracy": mean_accuracy, "estimated_task_score": task_score(mean_psnr, mean_accuracy), "frames": len(psnrs)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_root")
    parser.add_argument("validation_root")
    args = parser.parse_args()
    print(evaluate(args.train_root, args.validation_root))
