import numpy as np

from tinyworlds_viewer.visuals import contact_sheet, frame_delta, frame_stats, sample_indices


def test_sample_indices_are_bounded_and_sorted() -> None:
    assert sample_indices(10, 4) == [0, 3, 6, 9]
    assert sample_indices(10, 99, 7, 20) == [7, 8, 9]


def test_frame_stats_and_delta() -> None:
    black = np.zeros((2, 2, 3), dtype=np.uint8)
    white = np.full((2, 2, 3), 255, dtype=np.uint8)

    stats = frame_stats(white)
    assert stats.brightness == 255
    assert stats.unique_colors == 1

    delta = frame_delta(black, white)
    assert delta.shape == white.shape
    assert delta.max() == 255


def test_contact_sheet_dimensions() -> None:
    frames = [np.zeros((4, 5, 3), dtype=np.uint8) for _ in range(3)]
    sheet = contact_sheet(frames, columns=2, scale=2, pad=1)
    assert sheet.size == (23, 55)
