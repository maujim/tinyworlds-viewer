from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

DEFAULT_CACHE_DIR = Path("~/.cache/huggingface/hub/datasets--AlmondGod--tinyworlds").expanduser()


@dataclass(frozen=True)
class TinyWorldsFile:
    name: str
    path: Path
    frame_count: int
    height: int
    width: int
    channels: int

    @property
    def world(self) -> str:
        return self.name.removesuffix("_frames.h5")


class TinyWorldsDataset:
    """Thin HDF5 helper. Keeps the scaffold useful without prescribing viewer design."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or discover_dataset()
        self.files = scan_files(self.root)
        if not self.files:
            raise FileNotFoundError(f"No *_frames.h5 files found below {self.root}")

    def get_file(self, world: str) -> TinyWorldsFile:
        for file in self.files:
            if file.world == world or file.name == world:
                return file
        choices = ", ".join(file.world for file in self.files)
        raise KeyError(f"Unknown world {world!r}. Choices: {choices}")

    def read_frame(self, world: str, index: int) -> np.ndarray:
        file = self.get_file(world)
        if index < 0 or index >= file.frame_count:
            raise IndexError(f"Frame index {index} out of range 0..{file.frame_count - 1}")
        with h5py.File(file.path, "r") as handle:
            frame = handle["frames"][index]
        return np.asarray(frame, dtype=np.uint8)

    def read_frames(self, world: str, indices: list[int]) -> list[np.ndarray]:
        return [self.read_frame(world, index) for index in indices]


def discover_dataset() -> Path:
    """Find the local Hugging Face cache snapshot for AlmondGod/tinyworlds.

    Override with TINYWORLDS_DATASET_DIR if copied elsewhere.
    """
    env_path = os.environ.get("TINYWORLDS_DATASET_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()

    snapshots = DEFAULT_CACHE_DIR / "snapshots"
    if snapshots.exists():
        candidates = sorted(p for p in snapshots.iterdir() if p.is_dir())
        for candidate in reversed(candidates):
            if list(candidate.glob("*_frames.h5")):
                return candidate.resolve()

    return DEFAULT_CACHE_DIR.resolve()


def scan_files(root: Path) -> list[TinyWorldsFile]:
    files: list[TinyWorldsFile] = []
    for path in sorted(root.glob("*_frames.h5")):
        with h5py.File(path, "r") as handle:
            frames = handle["frames"]
            count, height, width, channels = frames.shape
        files.append(
            TinyWorldsFile(
                name=path.name,
                path=path,
                frame_count=int(count),
                height=int(height),
                width=int(width),
                channels=int(channels),
            )
        )
    return files
