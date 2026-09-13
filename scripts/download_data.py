"""Download and extract the checked-in official task archives."""

import tarfile
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/OlimpiadaAI/II-OlimpiadaAI/main/3_etap/1_inpainting"


if __name__ == "__main__":
    root = Path("data")
    root.mkdir(exist_ok=True)
    for split in ("train", "val"):
        archive = root / f"{split}.tar.gz"
        urllib.request.urlretrieve(f"{BASE}/{split}.tar.gz", archive)
        destination = root / split
        destination.mkdir(exist_ok=True)
        with tarfile.open(archive) as source:
            source.extractall(destination, filter="data")
