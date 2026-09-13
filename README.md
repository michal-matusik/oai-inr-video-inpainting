# Reconstructed INR Video Inpainting - Polish AI Olympiad II Final

This repository is a reconstructed reference solution for the final-stage INR video inpainting task in the Polish Artificial Intelligence Olympiad.
It is not the author's original competition submission.

The solution exposes missing video pixels as a function of normalized x, y, and time coordinates.
Its validated baseline reconstructs unseen timestamps by interpolating the nearest observed RGB and mask fields.
The repository also includes a SIREN-style neural field for follow-up experiments.

![Implicit neural representation illustration](assets/task-inr-diagram.jpg)

*INR illustration referenced by the official Polish AI Olympiad II final notebook.*

## Quick start

Install `requirements.txt`, download the official archives with `python scripts/download_data.py`, then run:

```bash
python -m src.evaluate data/train data/val
python -m unittest discover -s tests -v
```

## Validation

The reconstruction was evaluated on the ten held-out frames in the official public validation archive.

| Metric | Public validation result |
| :-- | --: |
| Mean PSNR | 29.329 dB |
| Mask accuracy | 95.483% |
| Estimated official score | 95/100 |

The run uses only the official training frames to construct the field.
Validation images and masks are used only for measurement.
See `SOLUTION.md` for the method, limitations, environment, and continuation target.

## Provenance

The original Polish task notebook is retained as `notebooks/original_submission.ipynb`.
`docs/task_en.md` is an English task summary.
