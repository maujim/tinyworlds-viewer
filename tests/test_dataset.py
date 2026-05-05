from pathlib import Path

import h5py
import numpy as np

from tinyworlds_viewer.dataset import TinyWorldsDataset, scan_files


def test_scan_files_and_read_frame(tmp_path: Path) -> None:
    path = tmp_path / "toy_frames.h5"
    data = np.zeros((3, 4, 5, 3), dtype=np.uint8)
    data[1, :, :, 0] = 255
    with h5py.File(path, "w") as handle:
        handle.create_dataset("frames", data=data)

    files = scan_files(tmp_path)
    assert len(files) == 1
    assert files[0].world == "toy"
    assert files[0].frame_count == 3
    assert files[0].height == 4
    assert files[0].width == 5

    dataset = TinyWorldsDataset(tmp_path)
    frame = dataset.read_frame("toy", 1)
    assert frame.shape == (4, 5, 3)
    assert frame.dtype == np.uint8
    assert frame[:, :, 0].max() == 255
