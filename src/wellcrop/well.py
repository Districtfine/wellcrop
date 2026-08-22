"""Well representation holding geometry, labels, metadata, and cropped image data."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np


@dataclass
class Well:
    """Represents a single isolated well extracted from a plate scan.

    Attributes:
        x: Center x coordinate in the original full-scan image (pixels).
        y: Center y coordinate in the original full-scan image (pixels).
        radius: Well radius in pixels (full-res).
        label: Standard identifier (e.g. 'A1', 'B3').
        plate_letter: Plate identifier (e.g. 'A', 'B').
        row: 0-indexed row index within the plate grid.
        col: 0-indexed column index within the plate grid.
        locked: Whether Hough circles successfully locked onto the physical rim.
        crop_radius: Radius used for the cropped sub-image after rim trimming.
        image: Circularly masked (or raw) cropped RGB image array, if extracted.
        mask: Circular binary mask (uint8, 255 inside circle, 0 outside).
    """

    x: int
    y: int
    radius: int
    label: str
    plate_letter: str = "A"
    row: int = 0
    col: int = 0
    locked: bool = False
    crop_radius: int = 0
    image: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.crop_radius == 0 and self.radius > 0:
            self.crop_radius = int(self.radius * 0.94)

    @property
    def center(self) -> Tuple[int, int]:
        """(x, y) center pixel coordinates."""
        return self.x, self.y

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """(x_min, y_min, x_max, y_max) bounding box in full scan coordinates."""
        r = self.crop_radius or self.radius
        return self.x - r, self.y - r, self.x + r, self.y + r

    def extract_crop(
        self,
        scan_image: np.ndarray,
        rim_shrink: float = 0.94,
        mask_background: bool = True,
    ) -> np.ndarray:
        """Extract and mask this well from the full scan image.

        Args:
            scan_image: Full scan image (H, W, 3) or (H, W).
            rim_shrink: Fraction of physical radius to keep (shaves off outer plastic rim).
            mask_background: If True, blacks out the corners outside the well circle.

        Returns:
            Extracted well crop as a NumPy array.
        """
        img_rgb = (
            cv2.cvtColor(scan_image, cv2.COLOR_GRAY2RGB)
            if scan_image.ndim == 2
            else scan_image[..., :3]
        )
        crop_r = int(self.radius * rim_shrink)
        self.crop_radius = crop_r

        img_h, img_w = img_rgb.shape[:2]
        y_min, y_max = max(0, self.y - crop_r), min(img_h, self.y + crop_r)
        x_min, x_max = max(0, self.x - crop_r), min(img_w, self.x + crop_r)

        raw_crop = img_rgb[y_min:y_max, x_min:x_max].copy()

        # Create centered circular mask
        mask = np.zeros(raw_crop.shape[:2], dtype=np.uint8)
        cv2.circle(
            mask,
            (raw_crop.shape[1] // 2, raw_crop.shape[0] // 2),
            crop_r,
            255,
            -1,
        )
        self.mask = mask

        if mask_background:
            final_crop = cv2.bitwise_and(raw_crop, raw_crop, mask=mask)
        else:
            final_crop = raw_crop

        self.image = final_crop
        return final_crop

    def save(self, filepath: str) -> None:
        """Save the extracted well image to disk.

        Args:
            filepath: Destination file path (.png, .tif, .jpg).
        """
        if self.image is None:
            raise ValueError(
                f"Well {self.label} has no cropped image. Call extract_crop() first."
            )
        # OpenCV expects BGR for imwrite
        bgr = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, bgr)

    def as_dict(self) -> Dict[str, Any]:
        """Convert geometry and metadata to a plain dictionary."""
        return {
            "label": self.label,
            "plate": self.plate_letter,
            "x": self.x,
            "y": self.y,
            "radius": self.radius,
            "crop_radius": self.crop_radius,
            "row": self.row,
            "col": self.col,
            "locked": self.locked,
            **self.metadata,
        }
