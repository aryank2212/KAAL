"""
adversarial_patch.py — Adversarial patch generation, loading, and injection.

Supports three patch types:
  - NOISE:   Random pixel noise (universal perturbation)
  - PATTERN: Structured high-frequency checkerboard / grid pattern
  - LOADED:  Pre-made adversarial patch loaded from an image file

Patches are alpha-blended onto objects within their bounding boxes,
preserving label integrity.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from config import PatchType


class AdversarialPatchInjector:
    """
    Generates or loads adversarial patches and applies them to images
    at object bounding-box locations.
    """

    def __init__(
        self,
        patch_type: PatchType = PatchType.NOISE,
        scale_range: Tuple[float, float] = (0.1, 0.3),
        alpha: float = 0.8,
        patch_image_path: Optional[str] = None,
    ):
        """
        Args:
            patch_type:       Which generation method to use.
            scale_range:      Min/max patch size as a fraction of the bounding-box dimension.
            alpha:            Blending opacity (1.0 = fully opaque patch).
            patch_image_path: File path for PatchType.LOADED.
        """
        self.patch_type = patch_type
        self.scale_range = scale_range
        self.alpha = alpha
        self.loaded_patch: Optional[np.ndarray] = None

        if patch_type == PatchType.LOADED:
            if patch_image_path is None:
                raise ValueError("patch_image_path required when patch_type is LOADED")
            self.loaded_patch = cv2.imread(patch_image_path)
            if self.loaded_patch is None:
                raise FileNotFoundError(f"Could not load patch image: {patch_image_path}")

    # ── Patch Generation ──────────────────────────────────────────────────

    def generate_patch(self, height: int, width: int) -> np.ndarray:
        """
        Create a patch of the given size based on the configured patch type.

        Args:
            height: Patch height in pixels.
            width:  Patch width in pixels.

        Returns:
            BGR uint8 numpy array of shape (height, width, 3).
        """
        if self.patch_type == PatchType.NOISE:
            return self._generate_noise_patch(height, width)
        elif self.patch_type == PatchType.PATTERN:
            return self._generate_pattern_patch(height, width)
        elif self.patch_type == PatchType.LOADED:
            return self._resize_loaded_patch(height, width)
        else:
            raise ValueError(f"Unknown patch type: {self.patch_type}")

    @staticmethod
    def _generate_noise_patch(h: int, w: int) -> np.ndarray:
        """Random uniform noise — acts as a universal perturbation."""
        return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    @staticmethod
    def _generate_pattern_patch(h: int, w: int) -> np.ndarray:
        """
        High-frequency checkerboard pattern with color variation.
        Designed to trigger high-frequency anomaly detectors.
        """
        patch = np.zeros((h, w, 3), dtype=np.uint8)
        # Checkerboard cell size (small = high frequency)
        cell = max(2, min(h, w) // 8)
        for y in range(0, h, cell):
            for x in range(0, w, cell):
                if ((y // cell) + (x // cell)) % 2 == 0:
                    # Random bright color per cell
                    color = np.random.randint(150, 256, 3).tolist()
                    patch[y:y + cell, x:x + cell] = color
                else:
                    color = np.random.randint(0, 100, 3).tolist()
                    patch[y:y + cell, x:x + cell] = color
        return patch

    def _resize_loaded_patch(self, h: int, w: int) -> np.ndarray:
        """Resize pre-loaded patch to the target dimensions."""
        return cv2.resize(self.loaded_patch, (w, h), interpolation=cv2.INTER_LINEAR)

    # ── Patch Application ─────────────────────────────────────────────────

    def apply_patch(
        self,
        image: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
        prob: float = 0.5,
    ) -> Tuple[np.ndarray, bool]:
        """
        Probabilistically inject adversarial patches onto objects in the image.

        Args:
            image:  BGR uint8 image (H, W, 3).
            bboxes: List of bounding boxes as (x_center, y_center, w, h) in pixel coords.
            prob:   Probability of applying a patch to each bbox.

        Returns:
            Tuple of (modified_image, was_patched).
            Bounding box labels are NOT altered — the patch is placed *within*
            the existing bbox so annotations remain valid.
        """
        if len(bboxes) == 0:
            return image.copy(), False

        result = image.copy()
        patched = False

        for (cx, cy, bw, bh) in bboxes:
            if np.random.random() > prob:
                continue

            # Determine patch size as a random fraction of the bbox
            scale = np.random.uniform(*self.scale_range)
            patch_h = max(4, int(bh * scale))
            patch_w = max(4, int(bw * scale))

            # Generate the patch
            patch = self.generate_patch(patch_h, patch_w)

            # Random position WITHIN the bounding box
            x_min = int(cx - bw / 2)
            y_min = int(cy - bh / 2)
            x_max = int(cx + bw / 2) - patch_w
            y_max = int(cy + bh / 2) - patch_h

            if x_max <= x_min or y_max <= y_min:
                # Bbox too small for a patch at this scale — skip
                continue

            px = np.random.randint(x_min, x_max)
            py = np.random.randint(y_min, y_max)

            # Clip to image boundaries
            img_h, img_w = result.shape[:2]
            px = np.clip(px, 0, img_w - patch_w)
            py = np.clip(py, 0, img_h - patch_h)

            # Alpha-blend the patch onto the image
            roi = result[py:py + patch_h, px:px + patch_w]
            blended = cv2.addWeighted(patch, self.alpha, roi, 1 - self.alpha, 0)
            result[py:py + patch_h, px:px + patch_w] = blended
            patched = True

        return result, patched


def parse_yolo_labels(
    label_path: str, img_w: int, img_h: int
) -> Tuple[List[int], List[Tuple[int, int, int, int]]]:
    """
    Parse a YOLO-format label file into pixel-coordinate bounding boxes.

    YOLO format per line: class_id  x_center  y_center  width  height
    (all values normalized to [0, 1])

    Args:
        label_path: Path to the .txt label file.
        img_w:      Image width in pixels.
        img_h:      Image height in pixels.

    Returns:
        (class_ids, bboxes) where bboxes are (cx, cy, w, h) in pixel coords.
    """
    class_ids = []
    bboxes = []
    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx = float(parts[1]) * img_w
                cy = float(parts[2]) * img_h
                w = float(parts[3]) * img_w
                h = float(parts[4]) * img_h
                class_ids.append(cls_id)
                bboxes.append((int(cx), int(cy), int(w), int(h)))
    except FileNotFoundError:
        pass  # No labels for this image — return empty
    return class_ids, bboxes
