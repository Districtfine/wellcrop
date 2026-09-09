"""wellcrop: A computer vision utility for automated detection, alignment, and extraction of wells from multi-well plate scans."""

from .detector import PlateDetector, detect_wells_from_rois
from .edges import (
    detect_plate_rect_edges,
    reconcile_vertical_borders,
    resolve_axis,
    snap_edge,
)
from .geometry import pad_box, roi_frac_to_px, well_grid_fracs
from .grid import (
    LABEL_SCHEMES,
    fit_grid_axis,
    fit_grid_transform,
    place_wells,
    well_number,
)
from .refine import refine_well
from .visualization import draw_overlay, render_overlay_matplotlib
from .well import Well

__version__ = "0.2.0"

__all__ = [
    "PlateDetector",
    "Well",
    "detect_wells_from_rois",
    "draw_overlay",
    "render_overlay_matplotlib",
    "refine_well",
    "place_wells",
    "well_number",
    "LABEL_SCHEMES",
    "fit_grid_axis",
    "fit_grid_transform",
    "detect_plate_rect_edges",
    "snap_edge",
    "resolve_axis",
    "reconcile_vertical_borders",
    "roi_frac_to_px",
    "pad_box",
    "well_grid_fracs",
]
