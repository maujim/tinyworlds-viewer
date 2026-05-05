from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class FrameStats:
    brightness: float
    contrast: float
    red: float
    green: float
    blue: float
    unique_colors: int


def sample_indices(
    frame_count: int,
    count: int,
    start: int = 0,
    stop: int | None = None,
) -> list[int]:
    """Return evenly spaced frame indices, inclusive of the requested range ends."""
    if frame_count <= 0:
        return []
    stop = frame_count - 1 if stop is None else min(stop, frame_count - 1)
    start = max(0, min(start, stop))
    count = max(1, min(count, stop - start + 1))
    return sorted({int(round(x)) for x in np.linspace(start, stop, count)})


def frame_stats(frame: np.ndarray) -> FrameStats:
    pixels = frame.reshape(-1, frame.shape[-1]).astype(np.float32)
    luminance = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
    unique = np.unique(pixels.astype(np.uint8), axis=0).shape[0]
    return FrameStats(
        brightness=float(luminance.mean()),
        contrast=float(luminance.std()),
        red=float(pixels[:, 0].mean()),
        green=float(pixels[:, 1].mean()),
        blue=float(pixels[:, 2].mean()),
        unique_colors=int(unique),
    )


def frame_delta(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Per-pixel absolute RGB difference boosted for easy inspection."""
    delta = np.abs(current.astype(np.int16) - previous.astype(np.int16))
    return np.clip(delta * 3, 0, 255).astype(np.uint8)


def contact_sheet(
    frames: list[np.ndarray],
    labels: list[str] | None = None,
    columns: int = 6,
    scale: int = 2,
    pad: int = 8,
) -> Image.Image:
    """Build a labeled image grid from RGB numpy frames."""
    if not frames:
        raise ValueError("contact_sheet needs at least one frame")
    columns = max(1, columns)
    scale = max(1, scale)
    labels = labels or ["" for _ in frames]
    height, width = frames[0].shape[:2]
    tile_w, tile_h = width * scale, height * scale
    label_h = 18
    rows = math.ceil(len(frames) / columns)
    sheet = Image.new(
        "RGB",
        (columns * tile_w + (columns + 1) * pad, rows * (tile_h + label_h) + (rows + 1) * pad),
        (16, 20, 30),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, frame in enumerate(frames):
        row, col = divmod(i, columns)
        x = pad + col * (tile_w + pad)
        y = pad + row * (tile_h + label_h + pad)
        image = Image.fromarray(frame).resize((tile_w, tile_h), Image.Resampling.NEAREST)
        sheet.paste(image, (x, y))
        draw.text((x + 3, y + tile_h + 3), labels[i], fill=(220, 230, 255), font=font)
    return sheet
