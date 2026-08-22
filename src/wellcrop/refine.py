"""Per-well circle refinement using localized Hough transforms on downscaled ROIs."""

from typing import Tuple
import cv2
import numpy as np


def refine_well(
    gray: np.ndarray,
    cx: int,
    cy: int,
    r: int,
    search_frac: float = 0.5,
    radius_tol: float = 0.2,
    max_shift: float = 0.5,
    downscale_px: int = 300,
    hough_param1: int = 100,
    hough_param2: int = 30,
) -> Tuple[int, int, int, bool]:
    """Snap an analytic (cx, cy, r) to the physical well ring with a local Hough search.

    gray is the full-image grayscale (or CLAHE-enhanced grayscale). Runs Hough on a
    downscaled ROI for ~12x speedup on giant scans, then scales the result back up.

    Args:
        gray: Grayscale full scan image.
        cx: Approximate center x (in pixels).
        cy: Approximate center y (in pixels).
        r: Approximate radius (in pixels).
        search_frac: Fractional ROI search padding around (cx, cy).
        radius_tol: Allowed fractional radius deviation from seed radius.
        max_shift: Max allowed center shift (as fraction of radius) before rejecting.
        downscale_px: Maximum dimension for local ROI downscaling.
        hough_param1: Canny edge detector upper threshold for HoughCircles.
        hough_param2: Accumulator threshold for circle centers.

    Returns:
        (cx, cy, r, locked):
        locked is True if a physical ring was confirmed; False if no good ring was
        found (in which case the input cx, cy, r is returned unchanged).
    """
    pad = int(r * (1 + search_frac))
    height, width = gray.shape
    x0, x1 = max(0, cx - pad), min(width, cx + pad)
    y0, y1 = max(0, cy - pad), min(height, cy + pad)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return cx, cy, r, False

    roi = cv2.medianBlur(roi, 5)
    scale = min(1.0, downscale_px / max(roi.shape))
    search_roi = (
        cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else roi
    )

    circles = cv2.HoughCircles(
        search_roi,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(search_roi.shape),  # only expect one ring in this ROI
        param1=hough_param1,
        param2=hough_param2,
        minRadius=int(r * (1 - radius_tol) * scale),
        maxRadius=int(r * (1 + radius_tol) * scale),
    )
    if circles is None:
        return cx, cy, r, False
    if scale < 1.0:
        circles = circles / scale  # back to ROI (full-res) coordinates

    # pick the ring whose center is closest to the analytic prior (ROI center)
    roi_cx, roi_cy = cx - x0, cy - y0
    found_cx, found_cy, found_r = min(
        circles[0], key=lambda cir: (cir[0] - roi_cx) ** 2 + (cir[1] - roi_cy) ** 2
    )
    if (found_cx - roi_cx) ** 2 + (found_cy - roi_cy) ** 2 > (r * max_shift) ** 2:
        return cx, cy, r, False  # moved too far -> likely a colony or artifact, not this well's rim

    return int(x0 + found_cx), int(y0 + found_cy), int(found_r), True
