"""Tests for geometry calculations and edge/grid functions."""

import unittest
import numpy as np

from wellcrop.edges import resolve_axis, snap_edge
from wellcrop.geometry import pad_box, roi_frac_to_px, well_grid_fracs
from wellcrop.grid import fit_grid_axis, fit_grid_transform
from wellcrop.well import Well


class TestWellCropGeometry(unittest.TestCase):
    def test_well_grid_fracs(self):
        x_fracs, y_fracs = well_grid_fracs(rows=3, cols=2)
        self.assertEqual(len(x_fracs), 2)
        self.assertEqual(len(y_fracs), 3)
        self.assertTrue(np.isclose(x_fracs[0], 0.25))
        self.assertTrue(np.isclose(x_fracs[1], 0.75))
        self.assertTrue(np.isclose(y_fracs[0], 1 / 6))
        self.assertTrue(np.isclose(y_fracs[1], 3 / 6))
        self.assertTrue(np.isclose(y_fracs[2], 5 / 6))

    def test_roi_frac_to_px(self):
        roi = {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
        x0, y0, x1, y1 = roi_frac_to_px(roi, image_width=1000, image_height=2000)
        self.assertEqual((x0, y0, x1, y1), (100, 400, 400, 1200))

    def test_pad_box(self):
        box = (100, 200, 300, 400)
        # width=200, height=200, pad=20 (10%)
        padded = pad_box(box, image_width=1000, image_height=1000, margin_frac=0.10)
        self.assertEqual(padded, (80, 180, 320, 420))

    def test_resolve_axis(self):
        # Both edges confident and agree on size
        low_coord, high_coord, status = resolve_axis(
            low_snap=(50, 3.5),
            high_snap=(150, 4.0),
            prior_low=45,
            prior_high=145,
            size_tol=0.15,
        )
        self.assertEqual(status, "both")
        self.assertEqual((low_coord, high_coord), (50, 150))

    def test_fit_grid_axis(self):
        indices = [0, 1, 2]
        centers = [100.0, 200.0, 300.0]
        origin, pitch = fit_grid_axis(
            indices, centers, analytic_pitch=100.0, analytic_origin=100.0
        )
        self.assertTrue(np.isclose(pitch, 100.0))
        self.assertTrue(np.isclose(origin, 100.0))

    def test_well_extract_crop(self):
        # Synthetic 500x500 white canvas
        image = np.full((500, 500, 3), 255, dtype=np.uint8)
        well = Well(x=250, y=250, radius=50, label="A1")
        crop = well.extract_crop(image, rim_shrink=0.94, mask_background=True)

        self.assertEqual(crop.shape[0], 2 * int(50 * 0.94))
        self.assertEqual(crop.shape[1], 2 * int(50 * 0.94))
        # Corner should be blacked out by circular mask
        self.assertTrue(np.all(crop[0, 0] == [0, 0, 0]))
        # Center should remain white
        center_idx = crop.shape[0] // 2
        self.assertTrue(np.all(crop[center_idx, center_idx] == [255, 255, 255]))


if __name__ == "__main__":
    unittest.main()
