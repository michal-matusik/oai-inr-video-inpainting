# Solution notes

This is a retrospective reconstruction and is not the original competition submission.

The public training archive contains 59 observed frames from one 79-frame video.
The public validation archive contains ten held-out timestamps from the same sequence.
The implemented field stores the observed RGB and binary-mask values and linearly interpolates each missing timestamp from the nearest observed frame on either side.
The model exposes the coordinate-only interface required by the official notebook.

The train and validation archives are kept out of Git and can be obtained with `python scripts/download_data.py`.
Evaluation fits only from `data/train` and reads validation content only while computing metrics.

On an Apple Silicon CPU, the full public validation evaluation takes under two seconds after extraction.
The measured result is 29.329 dB mean PSNR and 95.483% mask accuracy across ten held-out frames.
Using the official scoring function, this corresponds to an estimated public-validation score of 95/100.

Temporal interpolation exceeds the full PSNR threshold but misses the 98% mask-accuracy threshold.
A continuation should replace the mask interpolation with a motion-compensated or learned space-time field and retain the same untouched validation protocol.
