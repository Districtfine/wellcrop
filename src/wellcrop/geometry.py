"""Geometry helpers for fractional bounding boxes and plate grid calculations."""

from typing import Dict, List, Tuple, Union


def well_grid_fracs(rows: int, cols: int) -> Tuple[List[float], List[float]]:
    """Even-spaced cell-center fractions inside a plate box (center of each grid cell).

    Args:
        rows: Number of well rows.
        cols: Number of well columns.

    Returns:
        Tuple of (x_fracs, y_fracs) representing fractional center positions in [0, 1].
    """
    x_fracs = [(col_index + 0.5) / cols for col_index in range(cols)]
    y_fracs = [(row_index + 0.5) / rows for row_index in range(rows)]
    return x_fracs, y_fracs


def roi_frac_to_px(
    roi: Dict[str, Union[float, int, str]],
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """Convert a fractional {x, y, w, h} box (each in [0,1]) to integer (x0, y0, x1, y1) pixels.

    Args:
        roi: Dictionary with keys 'x', 'y', 'w' (or 'width'), 'h' (or 'height').
        image_width: Full scan image width in pixels.
        image_height: Full scan image height in pixels.

    Returns:
        (x0, y0, x1, y1) bounding box in integer pixel coordinates.
    """
    x = float(roi.get("x", 0.0))
    y = float(roi.get("y", 0.0))
    w = float(roi.get("w", roi.get("width", 0.0)))
    h = float(roi.get("h", roi.get("height", 0.0)))

    x0 = int(round(x * image_width))
    y0 = int(round(y * image_height))
    x1 = int(round((x + w) * image_width))
    y1 = int(round((y + h) * image_height))
    return x0, y0, x1, y1


def pad_box(
    box_px: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    margin_frac: float = 0.20,
) -> Tuple[int, int, int, int]:
    """Expand a pixel box by margin_frac of its own width/height on each side, clamped.

    Args:
        box_px: (x0, y0, x1, y1) in pixels.
        image_width: Full image width in pixels.
        image_height: Full image height in pixels.
        margin_frac: Expansion margin as a fraction of box dimensions.

    Returns:
        (x0, y0, x1, y1) padded box clamped to [0, image_width/height].
    """
    x0, y0, x1, y1 = box_px
    pad_x = int(round((x1 - x0) * margin_frac))
    pad_y = int(round((y1 - y0) * margin_frac))
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(image_width, x1 + pad_x),
        min(image_height, y1 + pad_y),
    )
