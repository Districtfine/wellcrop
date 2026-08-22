"""Visualization and overlay generation for well detection QA."""

from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from .geometry import pad_box, roi_frac_to_px
from .well import Well


def draw_overlay(
    image: np.ndarray,
    wells: List[Union[Well, Tuple[int, int, int, str]]],
    roi_hints: Optional[List[Dict[str, Any]]] = None,
    plate_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    margin_frac: float = 0.20,
) -> np.ndarray:
    """Draw detection QA overlay on an RGB image using OpenCV.

    Renders:
        - Yellow rectangles: Padded ROI search areas.
        - Lime green rectangles: Detected rigid plate boundaries.
        - Red circles: Well outlines (Hough locked or grid fitted).
        - Yellow text: Well labels ('A1', 'B2', etc.).

    Args:
        image: Source RGB/Grayscale image array.
        wells: List of Well objects or (x, y, r, label) tuples.
        roi_hints: Optional list of fractional ROI hints to draw search areas.
        plate_boxes: Optional list of (x, y, w, h) detected plate boxes.
        margin_frac: Margin fraction used for ROI search box padding.

    Returns:
        RGB image array with annotations drawn.
    """
    img_rgb = (
        cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.ndim == 2
        else image[..., :3].copy()
    )
    h, w = img_rgb.shape[:2]

    # 1. Draw search boxes (yellow)
    if roi_hints:
        for roi in roi_hints:
            prior_box = roi_frac_to_px(roi, w, h)
            x0, y0, x1, y1 = pad_box(prior_box, w, h, margin_frac)
            cv2.rectangle(img_rgb, (x0, y0), (x1, y1), (255, 255, 0), 2)

    # 2. Draw detected plate boxes (lime green)
    if plate_boxes:
        for px, py, pw, ph in plate_boxes:
            cv2.rectangle(img_rgb, (px, py), (px + pw, py + ph), (0, 255, 0), 3)

    # 3. Draw well circles & labels
    for item in wells:
        if isinstance(item, Well):
            cx, cy, r, label = item.x, item.y, item.radius, item.label
        else:
            cx, cy, r, label = item[:4]

        # Well circle (red)
        cv2.circle(img_rgb, (int(cx), int(cy)), int(r), (255, 0, 0), 3)

        # Label text (yellow with dark shadow)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_pos = (int(cx) + 12, int(cy) + 12)
        cv2.putText(
            img_rgb, label, text_pos, font, 1.2, (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(
            img_rgb, label, text_pos, font, 1.2, (255, 255, 0), 2, cv2.LINE_AA
        )

    return img_rgb


def render_overlay_matplotlib(
    axis: Any,
    image_rgb: np.ndarray,
    roi_hints: List[Dict[str, Any]],
    wells: List[Union[Well, Tuple[int, int, int, str]]],
    title: str = "",
    plate_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    margin_frac: float = 0.20,
) -> None:
    """Render detection overlay onto a Matplotlib Axes object (useful for Notebooks).

    Args:
        axis: Matplotlib Axes instance.
        image_rgb: Source RGB image array.
        roi_hints: List of fractional ROI hints.
        wells: List of Well objects or (x, y, r, label) tuples.
        title: Plot title.
        plate_boxes: List of (x, y, w, h) detected plate bounding boxes.
        margin_frac: Margin fraction for search box padding.
    """
    import matplotlib.pyplot as plt

    axis.imshow(image_rgb)
    if title:
        axis.set_title(title)
    image_height, image_width = image_rgb.shape[:2]

    for roi in roi_hints:
        prior_box = roi_frac_to_px(roi, image_width, image_height)
        search_x0, search_y0, search_x1, search_y1 = pad_box(
            prior_box, image_width, image_height, margin_frac
        )
        axis.add_patch(
            plt.Rectangle(
                (search_x0, search_y0),
                search_x1 - search_x0,
                search_y1 - search_y0,
                fill=False,
                edgecolor="yellow",
                linewidth=1.5,
            )
        )

    if plate_boxes:
        for plate_x, plate_y, plate_width, plate_height in plate_boxes:
            axis.add_patch(
                plt.Rectangle(
                    (plate_x, plate_y),
                    plate_width,
                    plate_height,
                    fill=False,
                    edgecolor="lime",
                    linewidth=2,
                )
            )

    for item in wells:
        if isinstance(item, Well):
            center_x, center_y, radius, label = (
                item.x,
                item.y,
                item.radius,
                item.label,
            )
        else:
            center_x, center_y, radius, label = item[:4]

        axis.add_patch(
            plt.Circle(
                (center_x, center_y),
                radius,
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
        )
        axis.text(
            center_x + 10,
            center_y + 10,
            label,
            color="yellow",
            fontsize=12,
            fontweight="bold",
        )

    axis.set_aspect("equal")
