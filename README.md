# Reconstructed INR Video Inpainting - Polish AI Olympiad II Final

This repository is a reconstructed reference solution for the final-stage INR video inpainting task in the Polish Artificial Intelligence Olympiad.

The solution exposes missing video pixels as a function of normalized x, y, and time coordinates.
Its validated baseline reconstructs unseen timestamps by interpolating the nearest observed RGB and mask fields.
The repository also includes a SIREN-style neural field for follow-up experiments.

![Implicit neural representation illustration](assets/task-inr-diagram.jpg)

*INR illustration referenced by the official Polish AI Olympiad II final notebook.*

## Method

The task asks for an implicit neural representation (INR): a continuous function
`f(x, y, t) -> (RGB, mask)` over normalized pixel/time coordinates, queried at held-out
timestamps rather than interpolated in pixel space. The validated solution (`TemporalVideoField`
in `src/model.py`) is a deterministic, non-learned instance of this interface: for a queried
frame it returns the exact observed frame if one exists, otherwise linearly interpolates every
pixel's RGB and mask value between the two nearest *observed* frames bracketing it in time. This
works because the underlying video is a slowly-varying signal in time relative to the 79-frame
sequence length, so first-order (piecewise-linear) temporal interpolation is a strong baseline
without any spatial reasoning.

The repository also ships (but does not train by default) a genuine coordinate network,
`VideoINR`: a SIREN (Sitzmann et al., 2020) — an MLP with sine activations,
`phi(x) = sin(omega_0 * (W x + b))`, `omega_0 = 30` — plus a Fourier-feature positional encoding
helper (`fourier_features`, Tancik et al., 2020: `[sin, cos](2^k * pi * x)` for `k` frequency
bands) in `src/dataset.py`. Both address the well-known spectral bias of coordinate MLPs (a plain
ReLU/tanh MLP struggles to fit high-frequency spatial detail from low-dimensional coordinates
without such a reparameterization), and are kept for follow-up experiments — the reported score
below is from the deterministic baseline, not the SIREN.

The official score (`src/metrics.py: task_score`) splits 10 points for PSNR (linear ramp,
0 pts at 15.5 dB, full 10 pts at 23.5 dB) and 30 points for mask accuracy (linear ramp, 0 pts at
0.83, full 30 pts at 0.98). The baseline's 29.3 dB PSNR comfortably saturates the PSNR band, but
its 0.955 mask accuracy sits inside the accuracy ramp rather than past its 0.98 ceiling — temporal
interpolation reconstructs color well but blurs the (harder, high-frequency) mask boundary at
timestamps between observed frames, which a spatially-aware model would be expected to sharpen.

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
