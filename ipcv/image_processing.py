"""Small image-processing primitives used by the course labs.

The functions keep OpenCV at the edges: OpenCV is used when available, while
the deterministic NumPy/SciPy fallbacks keep CI and early lessons lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class Corner:
    """A detected 2D feature point with a simple response score."""

    x: float
    y: float
    response: float


def as_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a float grayscale image in the range [0, 1]."""

    array = np.asarray(image)
    if array.ndim == 2:
        gray = array.astype(np.float64)
    elif array.ndim == 3 and array.shape[2] >= 3:
        rgb = array[..., :3].astype(np.float64)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    else:
        raise ValueError("image must be HxW or HxWx3")

    if gray.size == 0:
        raise ValueError("image must not be empty")
    if gray.max() > 1.0:
        gray = gray / 255.0
    return gray


def sobel_edges(image: np.ndarray) -> np.ndarray:
    """Compute Sobel edge magnitude for a grayscale or RGB image."""

    gray = as_grayscale(image)
    dx = ndimage.sobel(gray, axis=1, mode="nearest")
    dy = ndimage.sobel(gray, axis=0, mode="nearest")
    return np.hypot(dx, dy)


def canny_edges(
    image: np.ndarray,
    *,
    low_threshold: float = 50.0,
    high_threshold: float = 150.0,
) -> np.ndarray:
    """Compute a binary edge map.

    OpenCV's Canny implementation is used when installed. The fallback is a
    Sobel magnitude threshold so tests can still exercise the workflow without
    pulling native OpenCV wheels into every environment.
    """

    gray = as_grayscale(image)
    try:
        import cv2
    except ImportError:
        magnitude = sobel_edges(gray)
        if magnitude.max() == 0:
            return np.zeros_like(gray, dtype=bool)
        threshold = high_threshold / 255.0
        return magnitude / magnitude.max() >= threshold

    uint8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    edges = cv2.Canny(uint8, low_threshold, high_threshold)
    return edges > 0


def detect_corners(
    image: np.ndarray,
    *,
    max_corners: int = 50,
    quality_level: float = 0.01,
    min_distance: int = 5,
) -> list[Corner]:
    """Detect stable 2D feature points for geometry exercises."""

    if max_corners <= 0:
        return []
    gray = as_grayscale(image)
    try:
        import cv2
    except ImportError:
        response = sobel_edges(gray)
        local_max = response == ndimage.maximum_filter(
            response,
            size=max(1, min_distance),
            mode="nearest",
        )
        threshold = response.max() * quality_level
        ys, xs = np.nonzero(local_max & (response >= threshold))
        scores = response[ys, xs]
        order = np.argsort(scores)[::-1][:max_corners]
        return [
            Corner(x=float(xs[index]), y=float(ys[index]), response=float(scores[index]))
            for index in order
        ]

    points = cv2.goodFeaturesToTrack(
        np.clip(gray * 255.0, 0, 255).astype(np.uint8),
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
    )
    if points is None:
        return []
    corners: list[Corner] = []
    for point in points.reshape(-1, 2):
        x, y = point
        corners.append(Corner(x=float(x), y=float(y), response=1.0))
    return corners
