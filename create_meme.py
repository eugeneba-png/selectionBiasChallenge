"""
Assemble the four-panel statistics meme (Reality → Estimate).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _as_gray_01(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 3:
        a = a[:, :, 0]
    if a.ndim != 2:
        raise ValueError(f"Expected 2D grayscale (or H×W×1), got shape {a.shape}")
    return np.clip(a, 0.0, 1.0)


def _resize_to_shape(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize grayscale [0,1] array to (height, width)."""
    u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil = Image.fromarray(u8, mode="L").resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(pil, dtype=np.float32) / 255.0


def _align_panel(img: np.ndarray, ref_h: int, ref_w: int) -> np.ndarray:
    g = _as_gray_01(img)
    if g.shape[0] == ref_h and g.shape[1] == ref_w:
        return g
    return _resize_to_shape(g, ref_h, ref_w)


def create_statistics_meme(
    original_img: np.ndarray,
    stipple_img: np.ndarray,
    block_letter_img: np.ndarray,
    masked_stipple_img: np.ndarray,
    output_path: str,
    dpi: int = 150,
    background_color: str = "white",
) -> None:
    """
    Build a 1×4 figure with labeled panels and save as PNG.

    Panel order: Reality, Your Model, Selection Bias, Estimate.
    If panel shapes differ from ``original_img``, panels are resized to match it.
    """
    ref_h, ref_w = _as_gray_01(original_img).shape
    panels = [
        _align_panel(original_img, ref_h, ref_w),
        _align_panel(stipple_img, ref_h, ref_w),
        _align_panel(block_letter_img, ref_h, ref_w),
        _align_panel(masked_stipple_img, ref_h, ref_w),
    ]
    titles = ["Reality", "Your Model", "Selection Bias", "Estimate"]

    n = len(panels)
    fig_w = max(12.0, 3.2 * n)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(fig_w, 4.25),
        constrained_layout=True,
    )
    fig.patch.set_facecolor(background_color)

    if n == 1:
        axes = [axes]

    for ax, data, title in zip(axes, panels, titles):
        ax.imshow(data, cmap="gray", vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_title(title, fontsize=12, fontweight="bold", color="#222222", pad=10)
        ax.set_facecolor(background_color)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#c8c8c8")
            spine.set_linewidth(1.0)

    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        dpi=dpi,
        facecolor=background_color,
        edgecolor="none",
        format="png",
    )
    plt.close(fig)
