"""Plate edge detection via directional gradient profile snapping and rigid axis resolution."""

from typing import Dict, List, Tuple
import cv2
import numpy as np


def snap_edge(
    strength_profile: np.ndarray,
    prior_index: int,
    radius: int,
    min_peak_ratio: float = 2.0,
) -> Tuple[int, float]:
    """Snap to the strong edge NEAREST the drawn (prior) edge within +/- radius.

    Since the user draws accurately on the true plate, the plate's own wall is the closest
    strong edge to the drawn line; a neighbouring plate's wall (or handwriting/labels to
    the side) is farther, so 'nearest strong edge' rejects it even when it has higher contrast.

    A column/row counts as an edge if its summed gradient exceeds min_peak_ratio x the
    profile median.

    Args:
        strength_profile: 1D array of summed gradient magnitudes across rows or columns.
        prior_index: Starting coordinate (from user's drawn ROI).
        radius: Search radius in pixels.
        min_peak_ratio: Minimum ratio over median gradient strength to qualify as an edge.

    Returns:
        (nearest_index, confidence):
        confidence is the chosen line's ratio to profile median (0.0 if nothing qualified).
    """
    profile_median = float(np.median(strength_profile)) or 1.0
    threshold = min_peak_ratio * profile_median
    low = max(0, prior_index - radius)
    high = min(len(strength_profile), prior_index + radius + 1)
    strong_indices = [
        index for index in range(low, high) if strength_profile[index] > threshold
    ]
    if not strong_indices:
        return prior_index, 0.0
    nearest = min(strong_indices, key=lambda index: abs(index - prior_index))
    return nearest, float(strength_profile[nearest]) / profile_median


def resolve_axis(
    low_snap: Tuple[int, float],
    high_snap: Tuple[int, float],
    prior_low: int,
    prior_high: int,
    size_tol: float = 0.15,
) -> Tuple[int, int, str]:
    """Combine the two snapped edges of one axis, treating the plate as rigid.

    A plate cannot change size between scans, it can only move. So both snapped edges are
    trusted only when the size they imply matches the drawn size within size_tol; otherwise
    at least one of them locked onto something that is not this plate's wall, and the axis is
    instead *translated* by whichever edge was more confident, keeping the drawn size. If
    neither edge was found the drawn axis is kept and reported as "none".

    Args:
        low_snap: (index, confidence) for the low side (left/top).
        high_snap: (index, confidence) for the high side (right/bottom).
        prior_low: Starting coordinate of low edge.
        prior_high: Starting coordinate of high edge.
        size_tol: Allowed fractional size disagreement before rejecting dual lock.

    Returns:
        (low_coord, high_coord, status) where status is "both", "low", "high", or "none".
    """
    low_index, low_confidence = low_snap
    high_index, high_confidence = high_snap
    prior_size = prior_high - prior_low

    if low_confidence and high_confidence:
        if abs((high_index - low_index) - prior_size) <= size_tol * prior_size:
            return low_index, high_index, "both"
    if low_confidence and low_confidence >= high_confidence:
        return low_index, low_index + prior_size, "low"
    if high_confidence:
        return high_index - prior_size, high_index, "high"
    return prior_low, prior_high, "none"


def detect_plate_rect_edges(
    gray: np.ndarray,
    search_box: Tuple[int, int, int, int],
    prior_box: Tuple[int, int, int, int],
    min_peak_ratio: float = 2.0,
    size_tol: float = 0.15,
) -> Tuple[Tuple[int, int, int, int], Dict[str, str]]:
    """Snap each of the four plate edges to the strong line NEAREST the drawn edge.

    Sums |gradient| along each edge direction (down columns for vertical left/right edges,
    across rows for horizontal top/bottom edges). Each edge is searched only within +/- the
    padding on that side and snapped to the nearest strong line. The two edges of an axis
    are then reconciled as a rigid plate.

    Args:
        gray: Grayscale full scan image.
        search_box: Padded search region (search_x0, search_y0, search_x1, search_y1).
        prior_box: Original drawn ROI (prior_x0, prior_y0, prior_x1, prior_y1).
        min_peak_ratio: Peak gradient threshold ratio.
        size_tol: Rigid plate dimension tolerance.

    Returns:
        ((x, y, w, h) in full-image pixels, {"x": status, "y": status}).
    """
    search_x0, search_y0, search_x1, search_y1 = search_box
    prior_x0, prior_y0, prior_x1, prior_y1 = prior_box
    crop = gray[search_y0:search_y1, search_x0:search_x1]

    gradient_x = np.abs(cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3))
    gradient_y = np.abs(cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3))
    column_strength = gradient_x.sum(axis=0)  # one value per column -> vertical edges
    row_strength = gradient_y.sum(axis=1)  # one value per row    -> horizontal edges

    # search radius on each side = the padding on that side (placement-shift tolerance)
    radius_left = max(1, prior_x0 - search_x0)
    radius_right = max(1, search_x1 - prior_x1)
    radius_top = max(1, prior_y0 - search_y0)
    radius_bottom = max(1, search_y1 - prior_y1)

    left_snap = snap_edge(
        column_strength, prior_x0 - search_x0, radius_left, min_peak_ratio
    )
    right_snap = snap_edge(
        column_strength, prior_x1 - search_x0, radius_right, min_peak_ratio
    )
    top_snap = snap_edge(
        row_strength, prior_y0 - search_y0, radius_top, min_peak_ratio
    )
    bottom_snap = snap_edge(
        row_strength, prior_y1 - search_y0, radius_bottom, min_peak_ratio
    )

    left_local, right_local, x_status = resolve_axis(
        left_snap, right_snap, prior_x0 - search_x0, prior_x1 - search_x0, size_tol
    )
    top_local, bottom_local, y_status = resolve_axis(
        top_snap, bottom_snap, prior_y0 - search_y0, prior_y1 - search_y0, size_tol
    )

    left_x = search_x0 + left_local
    right_x = search_x0 + right_local
    top_y = search_y0 + top_local
    bottom_y = search_y0 + bottom_local
    return (left_x, top_y, right_x - left_x, bottom_y - top_y), {
        "x": x_status,
        "y": y_status,
    }


def reconcile_vertical_borders(
    gray: np.ndarray,
    plate_boxes: List[Tuple[int, int, int, int]],
    prior_boxes: List[Tuple[int, int, int, int]],
    margin_frac: float = 0.05,
    min_peak_ratio: float = 2.0,
) -> List[Tuple[int, int, int, int]]:
    """Make vertically-stacked plates share one hard border, so they never gap or overlap.

    When plates sit in one molded tray with a single rib between them, detecting each
    plate's inner edge independently can produce slight gaps or overlaps on shifted scans.
    This finds the single strongest horizontal edge (the rib) in the band between their
    drawn inner edges and sets upper bottom = lower top = that line.

    Args:
        gray: Grayscale full scan image.
        plate_boxes: Detected plate boxes [(x, y, w, h), ...].
        prior_boxes: Drawn prior boxes [(x0, y0, x1, y1), ...].
        margin_frac: Search band fraction around drawn dividing line.
        min_peak_ratio: Threshold ratio to qualify as rib.

    Returns:
        Updated [(x, y, w, h), ...] plate boxes.
    """
    order = sorted(range(len(plate_boxes)), key=lambda index: plate_boxes[index][1])
    boxes = [list(box) for box in plate_boxes]
    for upper_index, lower_index in zip(order, order[1:]):
        # shared horizontal extent of the two boxes (only look at columns they both cover)
        shared_x0 = max(boxes[upper_index][0], boxes[lower_index][0])
        shared_x1 = min(
            boxes[upper_index][0] + boxes[upper_index][2],
            boxes[lower_index][0] + boxes[lower_index][2],
        )
        if shared_x1 <= shared_x0:
            continue

        drawn_upper_bottom = prior_boxes[upper_index][3]
        drawn_lower_top = prior_boxes[lower_index][1]
        upper_drawn_height = (
            prior_boxes[upper_index][3] - prior_boxes[upper_index][1]
        )
        lower_drawn_height = (
            prior_boxes[lower_index][3] - prior_boxes[lower_index][1]
        )
        radius = int(max(upper_drawn_height, lower_drawn_height) * margin_frac)
        band_lo = max(0, min(drawn_upper_bottom, drawn_lower_top) - radius)
        band_hi = min(
            gray.shape[0], max(drawn_upper_bottom, drawn_lower_top) + radius
        )
        default_border_y = (drawn_upper_bottom + drawn_lower_top) // 2

        band = gray[band_lo:band_hi, shared_x0:shared_x1]
        if band.size == 0:
            border_y = default_border_y
        else:
            gradient_y = np.abs(cv2.Sobel(band, cv2.CV_64F, 0, 1, ksize=3))
            row_strength = gradient_y.sum(axis=1)
            profile_median = float(np.median(row_strength)) or 1.0
            peak = int(np.argmax(row_strength))
            border_y = (
                band_lo + peak
                if row_strength[peak] > min_peak_ratio * profile_median
                else default_border_y
            )

        # snap shared edge
        boxes[upper_index][3] = max(1, border_y - boxes[upper_index][1])
        lower_bottom = boxes[lower_index][1] + boxes[lower_index][3]
        boxes[lower_index][1] = border_y
        boxes[lower_index][3] = max(1, lower_bottom - border_y)

    return [tuple(box) for box in boxes]
