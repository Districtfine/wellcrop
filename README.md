# wellcrop

A lightweight computer vision utility for automated detection, alignment, and extraction of individual wells from multi-well plate scans.

---

## Highlights

* 🔍 **Shift & Skew Tolerant:** Accommodates hand-placed plates on flatbed scanners via directional gradient edge snapping and 2D affine grid estimation.
* 🎯 **Hough Ring Locking:** Snaps analytic well centers to physical plastic rims via localized Hough circle transforms.
* ✂️ **Automatic Masking:** Crops and isolates pure circular wells, shaving off outer plastic rims and blacking out corners.
* 🪶 **Zero Heavy AI Dependencies:** No PyTorch, CUDA, or model weights required. Built purely on `numpy`, `opencv-python`, and `scipy`.

---

## Installation

```bash
pip install wellcrop
```

---

## Quickstart

```python
import tifffile as tifi
import wellcrop

# 1. Load scan image
image = tifi.imread("plate_scan.tif")

# 2. Define ROI hints (fractions in [0, 1] drawn once on a reference scan)
roi_hints = [
    {"x": 0.05, "y": 0.08, "w": 0.40, "h": 0.84, "rows": 3, "cols": 2, "letter": "A"}
]

# 3. Detect & extract well crops
detector = wellcrop.PlateDetector(margin_frac=0.20)
wells = detector.crop(image, roi_hints=roi_hints)

# 4. Access individual well crops and metadata
for well in wells:
    print(f"Well {well.label}: center=({well.x}, {well.y}), radius={well.radius}px")
    # well.image is a circular-masked NumPy RGB array
    well.save(f"output/{well.label}.png")
```

---

## Visual QA Overlay

Generate diagnostic overlays (showing padded search boxes, detected plate boxes, and snapped well rings):

```python
import cv2

# Draw overlay directly onto the scan
overlay = wellcrop.draw_overlay(image, wells, roi_hints=roi_hints)
cv2.imwrite("grid_preview.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
```

---

## How it Works

1. **ROI-Scoped Search:** Expands the user's initial approximate plate bounding box by `margin_frac` to absorb scanner placement shift.
2. **Directional Edge Snapping:** Projects Sobel gradients along vertical and horizontal axes to find true plate walls, reconciling them under rigid geometry constraints.
3. **Stacked Plate Reconciliation:** Identifies shared dividing ribs for multi-tray formats so vertically stacked plates never overlap.
4. **Grid Estimation & Ring Snapping:** Computes analytic well centers, locks physical rims with local Hough circle searches, and fits a 2D affine transform to absorb plate rotation.
5. **Rim Trimming & Masking:** Shaves outer plastic rims and masks corners so downstream analyzers only see pure well contents.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
