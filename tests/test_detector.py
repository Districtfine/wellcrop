"""End-to-end detector test on a synthetic multi-well plate image."""

import unittest
import cv2
import numpy as np

from wellcrop import PlateDetector, draw_overlay


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


if __name__ == "__main__":
    unittest.main()
