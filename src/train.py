"""Construct the deterministic coordinate field from official training frames."""

from .dataset import load_observed_frames
from .model import TemporalVideoField


def train(train_root: str) -> TemporalVideoField:
    return TemporalVideoField(*load_observed_frames(train_root))
