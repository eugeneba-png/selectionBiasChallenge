"""
Step 4: Render a block letter (default "S") as a grayscale mask matching image size.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _candidate_font_paths() -> list[Path]:
    """Bold/heavy sans fonts likely to render a clear block letter."""
    return [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _candidate_font_paths():
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default()


def create_block_letter_s(
    height: int,
    width: int,
    letter: str = "S",
    font_size_ratio: float = 0.9,
) -> np.ndarray:
    """
    Draw a centered block letter on a white canvas matching ``(height, width)``.

    Parameters
    ----------
    height : int
        Output array height (rows).
    width : int
        Output array width (columns).
    letter : str
        Single character to render (default ``"S"``).
    font_size_ratio : float
        Starting font size as a fraction of ``min(height, width)`` before fit checks.

    Returns
    -------
    np.ndarray
        Shape ``(height, width)``, values in ``[0, 1]`` — black letter (0) on white (1).
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height=}, {width=}")
    text = (letter or "S")[:1] if letter else "S"
    short = min(height, width)
    margin = max(2, short // 50)

    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)

    max_start = max(12, int(short * font_size_ratio))
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None
    for size in range(max_start, 7, -2):
        candidate = _load_font(size)
        bbox = draw.textbbox((width // 2, height // 2), text, font=candidate, anchor="mm")
        left, top, right, bottom = bbox
        if (
            left >= margin - 1
            and top >= margin - 1
            and right <= width - margin + 1
            and bottom <= height - margin + 1
        ):
            font = candidate
            break

    if font is None:
        font = _load_font(max(12, short // 4))

    draw.text((width // 2, height // 2), text, fill=0, font=font, anchor="mm")

    out = np.asarray(img, dtype=np.float32) / 255.0
    return np.clip(out, 0.0, 1.0)
