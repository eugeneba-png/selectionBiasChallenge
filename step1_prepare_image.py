"""
Step 1: Prepare a black and white image for the statistics meme.
Loads an image, converts to grayscale, and resizes to appropriate dimensions
while maintaining aspect ratio.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from PIL import Image


def _demo_grayscale_pil() -> Image.Image:
    """Deterministic stand-in when no image file is available (e.g. first render)."""
    rng = np.random.default_rng(42)
    h, w = 600, 400
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)
    base = 0.3 + 0.4 * np.sin(4 * np.pi * xx) * np.cos(3 * np.pi * yy)
    base = base + 0.12 * rng.random((h, w))
    arr = (np.clip(base, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def prepare_image(
    img_path: str | Path,
    max_size: int = 512,
    target_size: tuple[int, int] | None = None
) -> np.ndarray:
    """
    Load an image, convert to grayscale, and resize to appropriate dimensions
    for the statistics meme while maintaining aspect ratio.
    
    Parameters
    ----------
    img_path : str | Path
        Path to the input image file. If the file is missing, a built-in demo
        pattern is used and a warning is shown (so notebooks still run).
    max_size : int
        Maximum dimension (width or height) if target_size is None.
        Image will be resized to fit within this size while maintaining aspect ratio.
    target_size : tuple[int, int] | None
        Optional target size (width, height). If provided, image will be resized
        to this size. If None, uses max_size to determine dimensions.
    
    Returns
    -------
    img_array : np.ndarray
        Grayscale image as 2D array (height, width) with values in [0, 1]
    """
    path = Path(img_path).expanduser()
    if not path.is_file():
        warnings.warn(
            f"Image not found: {path!s}. Using a built-in demo pattern. "
            f"Place your photo at that path, or set img_path to a file that exists.",
            UserWarning,
            stacklevel=2,
        )
        original_img = _demo_grayscale_pil()
    else:
        original_img = Image.open(path)
    
    # Convert to grayscale if needed
    if original_img.mode != 'L':
        original_img = original_img.convert('L')
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(original_img, dtype=np.float32) / 255.0
    
    # Resize if needed
    if target_size is not None:
        # Resize to exact target size
        new_size = target_size
        img_resized_pil = original_img.resize(new_size, Image.Resampling.LANCZOS)
        if img_resized_pil.mode != 'L':
            img_resized_pil = img_resized_pil.convert('L')
        img_resized = np.array(img_resized_pil, dtype=np.float32) / 255.0
        print(f"Resized image to target size: {img_resized.shape}")
    elif img_array.shape[0] > max_size or img_array.shape[1] > max_size:
        # Resize to fit within max_size while maintaining aspect ratio
        scale = max_size / max(img_array.shape[0], img_array.shape[1])
        new_size = (int(img_array.shape[1] * scale), int(img_array.shape[0] * scale))
        img_resized_pil = original_img.resize(new_size, Image.Resampling.LANCZOS)
        if img_resized_pil.mode != 'L':
            img_resized_pil = img_resized_pil.convert('L')
        img_resized = np.array(img_resized_pil, dtype=np.float32) / 255.0
        print(f"Resized image from {img_array.shape} to {img_resized.shape} for processing")
    else:
        img_resized = img_array.copy()
        print(f"Image size: {img_resized.shape} (no resizing needed)")
    
    # Ensure img_resized is 2D grayscale
    if len(img_resized.shape) > 2:
        img_resized = img_resized[:, :, 0]
    elif len(img_resized.shape) == 2:
        pass
    else:
        raise ValueError(f"Unexpected image shape: {img_resized.shape}")
    
    print(f"Final image shape: {img_resized.shape} (should be 2D for grayscale)")
    return img_resized

