"""
Blue noise stippling functions using a modified void-and-cluster algorithm.
"""

from __future__ import annotations

import numpy as np
from importance_map import compute_importance


def toroidal_gaussian_kernel(h: int, w: int, sigma: float):
    """
    Create a periodic (toroidal) 2D Gaussian kernel centered at (0,0).
    The toroidal property means the kernel wraps around at the edges,
    ensuring consistent repulsion behavior regardless of point location.
    
    Parameters:
    -----------
    h : int
        Height of the kernel (should match image height)
    w : int
        Width of the kernel (should match image width)
    sigma : float
        Standard deviation of the Gaussian (controls repulsion radius)
    
    Returns:
    --------
    kern : np.ndarray
        Normalized 2D Gaussian kernel with toroidal wrapping
    """
    y = np.arange(h)
    x = np.arange(w)
    # Compute toroidal distances (minimum distance considering wrapping)
    dy = np.minimum(y, h - y)[:, None]
    dx = np.minimum(x, w - x)[None, :]
    # Compute Gaussian
    kern = np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))
    s = kern.sum()
    if s > 0:
        kern /= s  # Normalize
    return kern


def void_and_cluster(
    input_img: np.ndarray,
    percentage: float = 0.08,
    sigma: float = 0.9,
    content_bias: float = 0.9,
    importance_img: np.ndarray | None = None,
    noise_scale_factor: float = 0.1,
):
    """
    Generate blue noise stippling pattern from input image using a modified
    void-and-cluster algorithm with content-weighted importance.
    
    Parameters:
    -----------
    input_img : np.ndarray
        Input image as 2D array (grayscale, normalized to [0, 1])
    percentage : float
        Percentage of pixels to stipple (0.0 to 1.0). Lower values (0.05-0.12)
        create sparser, more focused patterns.
    sigma : float
        Standard deviation of Gaussian kernel for repulsion (in pixels).
        Controls the minimum spacing between stipples.
    content_bias : float
        Scales the importance of image content in the energy field.
        Higher values (0.8-0.95) prioritize following the importance map;
        lower values allow more uniform spatial distribution.
    importance_img : np.ndarray | None
        Optional precomputed importance map (same shape as input).
        If None, importance is computed automatically from the input image.
    noise_scale_factor : float
        Scale factor for exploration noise (lower = crisper features, less exploration).
        Values typically range from 0.05 to 0.2.
    
    Returns:
    --------
    final_stipple : np.ndarray
        Binary stippling pattern (0.0 = black dot, 1.0 = white background)
    samples : np.ndarray
        Array of (y, x, intensity) tuples for each stipple point
    """
    I = np.clip(input_img, 0.0, 1.0)
    h, w = I.shape

    # Compute or use provided importance map
    if importance_img is None:
        importance = compute_importance(I)
    else:
        importance = np.clip(importance_img, 0.0, 1.0)

    # Create toroidal Gaussian kernel for repulsion
    kernel = toroidal_gaussian_kernel(h, w, sigma)

    # Initialize energy field: lower energy → more likely to be picked
    energy_current = -importance * content_bias

    # Stipple buffer: start with white background; selected points become black dots
    final_stipple = np.ones_like(I)
    samples = []

    # Helper function to roll kernel to an arbitrary position
    def energy_splat(y, x):
        """Get energy contribution by rolling the kernel to position (y, x)."""
        return np.roll(np.roll(kernel, shift=y, axis=0), shift=x, axis=1)

    # Number of points to select
    num_points = int(I.size * percentage)

    # Choose first point near center with minimal energy
    cy, cx = h // 2, w // 2
    r = min(20, h // 10, w // 10)
    ys = slice(max(0, cy - r), min(h, cy + r))
    xs = slice(max(0, cx - r), min(w, cx + r))
    region = energy_current[ys, xs]
    flat = np.argmin(region)
    y0 = flat // (region.shape[1]) + (cy - r)
    x0 = flat % (region.shape[1]) + (cx - r)

    # Place first point
    energy_current = energy_current + energy_splat(y0, x0)
    energy_current[y0, x0] = np.inf  # Prevent reselection
    samples.append((y0, x0, I[y0, x0]))
    final_stipple[y0, x0] = 0.0  # Black dot

    # Iteratively place remaining points
    for i in range(1, num_points):
        # Add exploration noise that decreases over time
        exploration = 1.0 - (i / num_points) * 0.5  # Decrease from 1.0 to 0.5
        noise = np.random.normal(0.0, noise_scale_factor * content_bias * exploration, size=energy_current.shape)
        energy_with_noise = energy_current + noise

        # Find position with minimum energy (with noise for exploration)
        pos_flat = np.argmin(energy_with_noise)
        y = pos_flat // w
        x = pos_flat % w

        # Add Gaussian splat to prevent nearby points from being selected
        energy_current = energy_current + energy_splat(y, x)
        energy_current[y, x] = np.inf  # Prevent reselection

        # Record the sample
        samples.append((y, x, I[y, x]))
        final_stipple[y, x] = 0.0  # Black dot

    return final_stipple, np.array(samples)

