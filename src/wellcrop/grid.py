"""Well grid placement, per-axis fitting, and affine transform estimation."""

from typing import List, Optional, Tuple
import numpy as np

from .geometry import well_grid_fracs
from .refine import refine_well


LABEL_SCHEMES = ("row-major", "column-major")
"""Well numbering orders: "row-major" numbers across each row first (the
historical default); "column-major" numbers down each column first, so on a
3x2 plate the left column is A1-A3 and the right column A4-A6."""


def validate_label_scheme(scheme: str) -> None:
    """Raise ValueError unless scheme is one of LABEL_SCHEMES."""
    if scheme not in LABEL_SCHEMES:
        raise ValueError(
            f"label_scheme must be one of {' | '.join(LABEL_SCHEMES)}, got {scheme!r}"
        )


def well_number(row: int, col: int, rows: int, cols: int, scheme: str) -> int:
    """1-based linear well number for grid cell (row, col) under a scheme.

    A scheme is a grid traversal order: wells are numbered in the order the
    scheme visits them, and place_wells returns them in ascending label order.
    """
    validate_label_scheme(scheme)
    if scheme == "row-major":
        return row * cols + col + 1
    return col * rows + row + 1


def fit_grid_axis(
    locked_indices: List[int],
    locked_centers: List[float],
    analytic_pitch: float,
    analytic_origin: float,
    pitch_tol: float = 0.25,
) -> Tuple[float, float]:
    """Fit one axis of the well grid — origin and pitch — from the wells that locked.

    Wells are moulded on a regular grid, so along either axis a center is
    origin + index * pitch: a two-parameter line fit over grid indices. Fitting it from
    locked wells stops the drawn box from dictating well spacing.

    Args:
        locked_indices: Grid indices along this axis (e.g. [0, 1, 2]).
        locked_centers: Corresponding pixel center coordinates.
        analytic_pitch: Expected pitch from the plate box dimensions.
        analytic_origin: Expected origin from the plate box coordinates.
        pitch_tol: Allowed fractional deviation of fitted pitch from analytic pitch.

    Returns:
        (origin, pitch) for this axis.
    """
    indices = np.asarray(locked_indices, dtype=float)
    centers = np.asarray(locked_centers, dtype=float)
    if len(indices) == 0:
        return analytic_origin, analytic_pitch

    recentered_origin = float(centers.mean() - indices.mean() * analytic_pitch)
    if len(np.unique(indices)) < 2:
        return recentered_origin, analytic_pitch

    pitch, origin = np.polyfit(indices, centers, 1)
    if not (1 - pitch_tol) <= pitch / analytic_pitch <= (1 + pitch_tol):
        return recentered_origin, analytic_pitch
    return float(origin), float(pitch)


def fit_grid_transform(
    locked_cols: List[int],
    locked_rows: List[int],
    locked_points: List[Tuple[float, float]],
    analytic_pitch_x: float,
    analytic_pitch_y: float,
    pitch_tol: float = 0.25,
    max_rotation_degrees: float = 8.0,
) -> Optional[np.ndarray]:
    """Least-squares affine mapping grid index (col, row) -> pixel center; None if unusable.

    Fitting one 2x3 transform over the locked wells absorbs rotation and skew (when a tray
    sits a degree or two off square in the scanner) instead of leaving it in residuals.

    Returns None (prompting fallback to per-axis fit) when locks do not span >= 2 rows and
    >= 2 cols, when pitch deviates past pitch_tol, or when rotation exceeds max_rotation_degrees.
    """
    cols = np.asarray(locked_cols, dtype=float)
    rows = np.asarray(locked_rows, dtype=float)
    points = np.asarray(locked_points, dtype=float)
    if len(points) < 3 or len(np.unique(cols)) < 2 or len(np.unique(rows)) < 2:
        return None

    design = np.column_stack([cols, rows, np.ones(len(cols))])
    solution, *_ = np.linalg.lstsq(design, points, rcond=None)
    transform = solution.T  # rows: [a, b, tx], [c, d, ty]

    column_step, row_step = transform[:, 0], transform[:, 1]
    pitch_x = float(np.hypot(*column_step))
    pitch_y = float(np.hypot(*row_step))
    if pitch_x <= 0 or pitch_y <= 0:
        return None
    if not (1 - pitch_tol) <= pitch_x / analytic_pitch_x <= (1 + pitch_tol):
        return None
    if not (1 - pitch_tol) <= pitch_y / analytic_pitch_y <= (1 + pitch_tol):
        return None

    column_rotation = np.degrees(np.arctan2(column_step[1], column_step[0]))
    row_rotation = np.degrees(np.arctan2(row_step[0], row_step[1]))
    if (
        abs(column_rotation) > max_rotation_degrees
        or abs(row_rotation) > max_rotation_degrees
    ):
        return None
    return transform


def place_wells(
    refine_gray: np.ndarray,
    plate_box: Tuple[int, int, int, int],
    rows: int,
    cols: int,
    plate_letter: str = "A",
    label_scheme: str = "row-major",
    refine: bool = True,
    radius_pitch_frac: float = 0.45,
    lock_trust_frac: float = 0.25,
    grid_pitch_tol: float = 0.25,
    max_rotation_degrees: float = 8.0,
    hough_param1: int = 100,
    hough_param2: int = 30,
) -> List[Tuple[int, int, int, str, int, int, bool]]:
    """Place wells on a fitted rows x cols grid, anchored by whichever wells lock onto a real ring.

    Wells are numbered 1..rows*cols in label_scheme traversal order and returned in
    ascending label order — "row-major" numbers across each row first (the analytic
    loop order), "column-major" numbers down each column first (on a 3x2 plate the
    left column is A1-A3, the right one A4-A6).

    Returns:
        List of tuples: (center_x, center_y, radius, label, row_idx, col_idx, locked)
    """
    plate_x, plate_y, plate_width, plate_height = plate_box
    x_fracs, y_fracs = well_grid_fracs(rows, cols)
    base_radius = int(
        radius_pitch_frac * min(plate_width / cols, plate_height / rows)
    )

    analytic_centers, labels, row_indices, col_indices = [], [], [], []
    for row_index, y_frac in enumerate(y_fracs):
        for col_index, x_frac in enumerate(x_fracs):
            analytic_centers.append(
                (plate_x + plate_width * x_frac, plate_y + plate_height * y_frac)
            )
            labels.append(
                f"{plate_letter}{well_number(row_index, col_index, rows, cols, label_scheme)}"
            )
            row_indices.append(row_index)
            col_indices.append(col_index)

    if not refine:
        results = [
            (
                int(center_x),
                int(center_y),
                base_radius,
                label,
                r_idx,
                c_idx,
                False,
            )
            for (center_x, center_y), label, r_idx, c_idx in zip(
                analytic_centers, labels, row_indices, col_indices
            )
        ]
        results.sort(
            key=lambda placed: well_number(placed[4], placed[5], rows, cols, label_scheme)
        )
        return results

    found_by_index = {}
    locked_points, locked_radii, locked_rows, locked_cols = [], [], [], []
    for index, (center_x, center_y) in enumerate(analytic_centers):
        found_x, found_y, found_r, locked = refine_well(
            refine_gray,
            int(center_x),
            int(center_y),
            base_radius,
            hough_param1=hough_param1,
            hough_param2=hough_param2,
        )
        if locked:
            found_by_index[index] = (found_x, found_y)
            locked_points.append((found_x, found_y))
            locked_radii.append(found_r)
            locked_rows.append(row_indices[index])
            locked_cols.append(col_indices[index])

    analytic_pitch_x = (
        plate_width * (x_fracs[1] - x_fracs[0]) if cols > 1 else plate_width
    )
    analytic_pitch_y = (
        plate_height * (y_fracs[1] - y_fracs[0]) if rows > 1 else plate_height
    )

    transform = (
        fit_grid_transform(
            locked_cols,
            locked_rows,
            locked_points,
            analytic_pitch_x,
            analytic_pitch_y,
            pitch_tol=grid_pitch_tol,
            max_rotation_degrees=max_rotation_degrees,
        )
        if locked_points
        else None
    )

    if transform is None:
        origin_x, pitch_x = fit_grid_axis(
            locked_cols,
            [point[0] for point in locked_points],
            analytic_pitch_x,
            analytic_centers[0][0],
            pitch_tol=grid_pitch_tol,
        )
        origin_y, pitch_y = fit_grid_axis(
            locked_rows,
            [point[1] for point in locked_points],
            analytic_pitch_y,
            analytic_centers[0][1],
            pitch_tol=grid_pitch_tol,
        )
        centers = np.array(
            [
                [
                    origin_x + col_index * pitch_x,
                    origin_y + row_index * pitch_y,
                ]
                for row_index, col_index in zip(row_indices, col_indices)
            ]
        )
    else:
        grid_indices = np.column_stack(
            [col_indices, row_indices, np.ones(len(row_indices))]
        )
        centers = grid_indices @ transform.T

    # Prefer each well's own rim over grid prediction if within trust_radius
    trust_radius = lock_trust_frac * min(analytic_pitch_x, analytic_pitch_y)
    for index, found_point in found_by_index.items():
        disagreement = float(
            np.hypot(
                found_point[0] - centers[index][0],
                found_point[1] - centers[index][1],
            )
        )
        if disagreement <= trust_radius:
            centers[index] = found_point

    radius = int(np.median(locked_radii)) if locked_radii else base_radius

    # Cap radius so two adjacent wells never overlap
    if len(centers) > 1:
        offsets = centers[:, None, :] - centers[None, :, :]
        distances = np.hypot(offsets[..., 0], offsets[..., 1])
        np.fill_diagonal(distances, np.inf)
        radius = min(radius, int(distances.min() // 2))

    results = []
    for (center_x, center_y), label, r_idx, c_idx, orig_idx in zip(
        centers, labels, row_indices, col_indices, range(len(labels))
    ):
        is_locked = orig_idx in found_by_index
        results.append(
            (
                int(center_x),
                int(center_y),
                radius,
                label,
                r_idx,
                c_idx,
                is_locked,
            )
        )

    results.sort(
        key=lambda placed: well_number(placed[4], placed[5], rows, cols, label_scheme)
    )
    return results
