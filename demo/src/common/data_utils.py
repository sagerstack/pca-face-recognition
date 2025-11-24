import os
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _read_pgm(filepath: Path) -> np.ndarray:
    """Minimal PGM reader supporting P2 and P5 formats."""
    with open(filepath, "rb") as f:
        header = f.readline().decode("ascii").strip()
        if header not in ("P2", "P5"):
            raise ValueError(f"Unsupported PGM format in {filepath}")

        # Skip comments
        line = f.readline().decode("ascii")
        while line.startswith("#"):
            line = f.readline().decode("ascii")

        width, height = map(int, line.strip().split())
        max_val = int(f.readline().decode("ascii").strip())

        if header == "P2":
            pixels: List[int] = []
            while len(pixels) < width * height:
                part = f.readline().decode("ascii").strip()
                if not part:
                    continue
                pixels.extend(map(int, part.split()))
            data = np.array(pixels[: width * height], dtype=np.float32)
        else:
            dtype = np.uint8 if max_val <= 255 else np.uint16
            data = np.frombuffer(f.read(width * height * dtype().nbytes), dtype=dtype).astype(
                np.float32
            )

        return data.reshape((height, width))


def load_att_faces(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load AT&T dataset from directory containing s1..s40 subfolders with .pgm files.

    Returns:
        images: float32 array (N, H, W) normalized to [0,1]
        labels: int array (N,) zero-based person ids
    """
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_dir}")

    images: List[np.ndarray] = []
    labels: List[int] = []

    subject_dirs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("s")])
    for subj in subject_dirs:
        try:
            subj_id = int(subj.name[1:]) - 1  # zero-based for UI
        except ValueError:
            continue
        for img_file in sorted(subj.glob("*.pgm")):
            img = _read_pgm(img_file)
            images.append(img / 255.0)
            labels.append(subj_id)

    if not images:
        raise ValueError(f"No PGM images found under {data_dir}")

    images_np = np.stack(images, axis=0).astype(np.float32)
    labels_np = np.array(labels, dtype=np.int32)
    return images_np, labels_np
