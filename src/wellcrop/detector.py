"""High-level plate detector and multi-well batch extractor."""

from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from .edges import detect_plate_rect_edges, reconcile_vertical_borders
from .geometry import pad_box, roi_frac_to_px
from .grid import place_wells, validate_label_scheme
from .well import Well


class PlateDetector:
    """Configurable multi-well plate detector with shift tolerance and Hough snap.

    Parameters:
        rows: Default number of well rows per plate (e.g. 3 for 6-well, 4 for 24-well).
        cols: Default number of well cols per plate (e.g. 2 for 6-well, 6 for 24-well).
        label_scheme: Well numbering order — "row-major" numbers across rows first
            (A1 A2 / A3 A4 / A5 A6 on a 3x2 plate; historical default), "column-major"
            numbers down each column first (left column A1-A3, right column A4-A6).
        margin_frac: Placement shift search tolerance around drawn ROI boxes (default: 0.20).
        clahe_clip: Contrast enhancement limit for faint plastic edges (default: 3.0).
        refine_wells: Snap analytic well centers to physical rings via Hough (default: True).
        radius_pitch_frac: Initial well radius as fraction of grid pitch (default: 0.45).
        plate_size_tol: Dimension tolerance for rigid plate edge matching (default: 0.15).
        grid_pitch_tol: Tolerance for affine/per-axis grid pitch estimation (default: 0.25).
        max_grid_rotation_degrees: Maximum allowable skew angle for grid (default: 8.0).
        lock_trust_frac: Distance fraction from grid to trust an individual ring lock (default: 0.25).
        rim_shrink: Fraction of physical radius to keep in crop (default: 0.94).
        hough_param1: Canny edge high threshold for circle detection (default: 100).
        hough_param2: Accumulator threshold for circle detection (default: 30).
    """

    def __init__(
        self,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        label_scheme: str = "row-major",
        margin_frac: float = 0.20,
        clahe_clip: float = 3.0,
        refine_wells: bool = True,
        radius_pitch_frac: float = 0.45,
        plate_size_tol: float = 0.15,
        grid_pitch_tol: float = 0.25,
        max_grid_rotation_degrees: float = 8.0,
        lock_trust_frac: float = 0.25,
        rim_shrink: float = 0.94,
        hough_param1: int = 100,
        hough_param2: int = 30,
    ):
        self.rows = rows
        self.cols = cols
        validate_label_scheme(label_scheme)
        self.label_scheme = label_scheme
        self.margin_frac = margin_frac
        self.clahe_clip = clahe_clip
        self.refine_wells = refine_wells
        self.radius_pitch_frac = radius_pitch_frac
        self.plate_size_tol = plate_size_tol
        self.grid_pitch_tol = grid_pitch_tol
        self.max_grid_rotation_degrees = max_grid_rotation_degrees
        self.lock_trust_frac = lock_trust_frac
        self.rim_shrink = rim_shrink
        self.hough_param1 = hough_param1
        self.hough_param2 = hough_param2

    def detect(
        self,
        image: np.ndarray,
        roi_hints: List[Dict[str, Any]],
        return_boxes: bool = False,
    ) -> Union[List[Well], Tuple[List[Well], List[Tuple[int, int, int, int]]]]:
        """Detect plates and locate all wells in full scan image coords.

        Args:
            image: Full scan image as a NumPy array (H, W, 3) or (H, W).
            roi_hints: List of ROI dicts [{'x': float, 'y': float, 'w': float, 'h': float, ...}].
            return_boxes: If True, also return detected [(x, y, w, h), ...] plate boxes.

        Returns:
            List of Well instances (and optionally detected plate boxes).
        """
        img_rgb = (
            cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            if image.ndim == 2
            else image[..., :3]
        )
        image_height, image_width = img_rgb.shape[:2]

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        refine_gray = (
            cv2.createCLAHE(
                clipLimit=self.clahe_clip, tileGridSize=(8, 8)
            ).apply(gray)
            if self.refine_wells
            else gray
        )

        # Pass 1: detect each plate box independently within its padded ROI
        prior_boxes = [
            roi_frac_to_px(roi, image_width, image_height) for roi in roi_hints
        ]
        plate_boxes = []
        for roi, prior_box in zip(roi_hints, prior_boxes):
            search_box = pad_box(
                prior_box, image_width, image_height, self.margin_frac
            )
            plate_box, status = detect_plate_rect_edges(
                gray,
                search_box,
                prior_box,
                size_tol=self.plate_size_tol,
            )
            plate_boxes.append(plate_box)

        # Pass 2: reconcile vertical borders for stacked plates
        if len(plate_boxes) > 1:
            plate_boxes = reconcile_vertical_borders(
                gray, plate_boxes, prior_boxes, self.margin_frac
            )

        # Pass 3: place and refine wells inside reconciled boxes
        all_wells: List[Well] = []
        for roi_index, (roi, plate_box) in enumerate(zip(roi_hints, plate_boxes)):
            plate_letter = roi.get("letter") or chr(ord("A") + roi_index)
            rows = roi.get("rows") or self.rows
            cols = roi.get("cols") or self.cols
            label_scheme = roi.get("label_scheme") or self.label_scheme

            if rows is None or cols is None:
                raise ValueError(
                    f"Plate layout (rows/cols) must be specified in PlateDetector or ROI hint {roi_index}"
                )

            placed = place_wells(
                refine_gray,
                plate_box,
                rows=rows,
                cols=cols,
                plate_letter=plate_letter,
                label_scheme=label_scheme,
                refine=self.refine_wells,
                radius_pitch_frac=self.radius_pitch_frac,
                lock_trust_frac=self.lock_trust_frac,
                grid_pitch_tol=self.grid_pitch_tol,
                max_rotation_degrees=self.max_grid_rotation_degrees,
                hough_param1=self.hough_param1,
                hough_param2=self.hough_param2,
            )

            for (
                cx,
                cy,
                r,
                label,
                row_idx,
                col_idx,
                locked,
            ) in placed:
                well = Well(
                    x=cx,
                    y=cy,
                    radius=r,
                    label=label,
                    plate_letter=plate_letter,
                    row=row_idx,
                    col=col_idx,
                    locked=locked,
                )
                all_wells.append(well)

        if return_boxes:
            return all_wells, plate_boxes
        return all_wells

    def crop(
        self,
        image: np.ndarray,
        roi_hints: List[Dict[str, Any]],
        mask_background: bool = True,
    ) -> List[Well]:
        """Detect and immediately extract cropped image data for each well.

        Args:
            image: Full scan image as a NumPy array (H, W, 3) or (H, W).
            roi_hints: List of fractional ROI hints.
            mask_background: If True, blacks out pixels outside circular rim.

        Returns:
            List of Well objects with `.image` and `.mask` populated.
        """
        wells = self.detect(image, roi_hints, return_boxes=False)
        for well in wells:
            well.extract_crop(
                image,
                rim_shrink=self.rim_shrink,
                mask_background=mask_background,
            )
        return wells


def detect_wells_from_rois(
    image_rgb: np.ndarray,
    roi_hints: List[Dict[str, Any]],
    margin_frac: float = 0.20,
    clahe_clip: float = 3.0,
    refine: bool = True,
    return_boxes: bool = False,
    label_scheme: str = "row-major",
) -> Union[
    List[Tuple[int, int, int, str]],
    Tuple[List[Tuple[int, int, int, str]], List[Tuple[int, int, int, int]]],
]:
    """Compatibility wrapper matching original notebook signature.

    Returns:
        List of (x, y, radius, label) tuples (and optionally plate boxes).
    """
    detector = PlateDetector(
        margin_frac=margin_frac,
        clahe_clip=clahe_clip,
        refine_wells=refine,
        label_scheme=label_scheme,
    )
    res = detector.detect(image_rgb, roi_hints, return_boxes=return_boxes)
    if return_boxes:
        wells, boxes = res
        return [(w.x, w.y, w.radius, w.label) for w in wells], boxes
    wells = res
    return [(w.x, w.y, w.radius, w.label) for w in wells]
