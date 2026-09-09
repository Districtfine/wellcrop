"""End-to-end detector test on a synthetic multi-well plate image."""

import unittest
import cv2
import numpy as np

from wellcrop import PlateDetector, draw_overlay, well_number


class TestPlateDetector(unittest.TestCase):
    def setUp(self):
        # Create a synthetic 1000x800 image with 1 plate of 3x2 wells
        self.image = np.full((1000, 800, 3), 40, dtype=np.uint8)

        # Draw a white plate rectangle
        cv2.rectangle(self.image, (100, 100), (700, 900), (200, 200, 200), -1)

        # Draw 6 circular wells (3 rows, 2 cols)
        for r_idx, cy in enumerate([250, 500, 750]):
            for c_idx, cx in enumerate([250, 550]):
                # Outer ring
                cv2.circle(self.image, (cx, cy), 90, (100, 100, 100), 4)
                # Well interior
                cv2.circle(self.image, (cx, cy), 86, (240, 240, 240), -1)

    def test_detector_crop_pipeline(self):
        detector = PlateDetector(rows=3, cols=2, margin_frac=0.15)
        roi_hints = [
            {
                "x": 0.10,
                "y": 0.08,
                "w": 0.77,
                "h": 0.82,
                "rows": 3,
                "cols": 2,
                "letter": "A",
            }
        ]

        wells = detector.crop(self.image, roi_hints=roi_hints)

        self.assertEqual(len(wells), 6)
        labels = [w.label for w in wells]
        self.assertEqual(labels, ["A1", "A2", "A3", "A4", "A5", "A6"])

        for well in wells:
            self.assertIsNotNone(well.image)
            self.assertIsNotNone(well.mask)
            self.assertGreater(well.radius, 50)
            self.assertEqual(well.image.shape[:2], (2 * well.crop_radius, 2 * well.crop_radius))

        # Test overlay generation
        overlay = draw_overlay(self.image, wells, roi_hints=roi_hints)
        self.assertEqual(overlay.shape, self.image.shape)



class TestWellNumber(unittest.TestCase):
    def test_row_major_counts_across_rows(self):
        seq = [well_number(r, c, 2, 3, "row-major") for r in range(2) for c in range(3)]
        self.assertEqual(seq, [1, 2, 3, 4, 5, 6])

    def test_column_major_counts_down_columns(self):
        seq = [well_number(r, c, 2, 3, "column-major") for r in range(2) for c in range(3)]
        # Down column 0 first (1, 2), then column 1 (3, 4), then column 2 (5, 6)
        self.assertEqual(seq, [1, 3, 5, 2, 4, 6])

    def test_schemes_coincide_for_single_row_or_column(self):
        for rows, cols in ((3, 1), (1, 3)):
            row_major = [
                well_number(r, c, rows, cols, "row-major")
                for r in range(rows) for c in range(cols)
            ]
            column_major = [
                well_number(r, c, rows, cols, "column-major")
                for r in range(rows) for c in range(cols)
            ]
            self.assertEqual(row_major, column_major)

    def test_unknown_scheme_raises(self):
        with self.assertRaises(ValueError):
            well_number(0, 0, 3, 2, "snake")


class TestLabelSchemeDetection(unittest.TestCase):
    def setUp(self):
        self.image = np.full((1000, 800, 3), 40, dtype=np.uint8)
        cv2.rectangle(self.image, (100, 100), (700, 900), (200, 200, 200), -1)
        for r_idx, cy in enumerate([250, 500, 750]):
            for c_idx, cx in enumerate([250, 550]):
                cv2.circle(self.image, (cx, cy), 90, (100, 100, 100), 4)
                cv2.circle(self.image, (cx, cy), 86, (240, 240, 240), -1)
        self.roi_hints = [
            {"x": 0.10, "y": 0.08, "w": 0.77, "h": 0.82, "rows": 3, "cols": 2, "letter": "A"}
        ]

    def test_column_major_numbers_down_left_column_first(self):
        detector = PlateDetector(
            rows=3, cols=2, margin_frac=0.15, label_scheme="column-major"
        )
        wells = detector.detect(self.image, self.roi_hints)

        # Wells come back in ascending label order under every scheme
        self.assertEqual([w.label for w in wells], ["A1", "A2", "A3", "A4", "A5", "A6"])

        # The mapping is what changes: A2 is mid-left, A4 is top-right
        by_label = {w.label: w for w in wells}
        self.assertEqual((by_label["A2"].row, by_label["A2"].col), (1, 0))
        self.assertEqual((by_label["A4"].row, by_label["A4"].col), (0, 1))
        self.assertLess(by_label["A2"].x, by_label["A4"].x)
        self.assertGreater(by_label["A2"].y, by_label["A1"].y)

    def test_hint_label_scheme_overrides_detector(self):
        detector = PlateDetector(
            rows=3, cols=2, margin_frac=0.15, label_scheme="column-major"
        )
        roi_hints = [dict(self.roi_hints[0], label_scheme="row-major")]
        wells = detector.detect(self.image, roi_hints)
        by_label = {w.label: w for w in wells}
        # Detector default says column-major, hint wins: A2 is top-right again
        self.assertEqual((by_label["A2"].row, by_label["A2"].col), (0, 1))

    def test_unknown_label_scheme_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            PlateDetector(rows=3, cols=2, label_scheme="snake")


if __name__ == "__main__":
    unittest.main()
