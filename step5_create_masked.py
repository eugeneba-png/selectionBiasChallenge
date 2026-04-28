"""
Step 5: Apply block-letter mask to stippled image (biased estimate panel).
"""

from __future__ import annotations

import numpy as np


def create_masked_stipple(
    stipple_img: np.ndarray,
    mask_img: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Where the mask is dark (selection pattern), clear stipples to white; elsewhere keep stipples.

    Parameters
    ----------
    stipple_img : np.ndarray
        Stipple image, shape ``(H, W)``, values in ``[0, 1]`` (black dots, white background).
    mask_img : np.ndarray
        Same shape; ``0`` = mask region (letter), ``1`` = keep region.
    threshold : float
        Pixels with ``mask_img < threshold`` are treated as mask (stipples removed → ``1.0``).

    Returns
    -------
    np.ndarray
        Same shape as inputs, ``float`` in ``[0, 1]``.
    """
    if stipple_img.shape != mask_img.shape:
        raise ValueError(
            f"Shape mismatch: stipple_img {stipple_img.shape} vs mask_img {mask_img.shape}"
        )
    if stipple_img.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got stipple_img.ndim={stipple_img.ndim}")

    s = np.asarray(stipple_img, dtype=np.float32)
    m = np.asarray(mask_img, dtype=np.float32)
    out = np.where(m < float(threshold), 1.0, s)
    return np.clip(out, 0.0, 1.0)
