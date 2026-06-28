"""
dual_stream_defense.py — FFT-based dual-stream adversarial defense.

Stream 1: Standard YOLO detection.
Stream 2: Frequency-domain anomaly detection using 2D FFT.

Fusion: If the high-frequency anomaly score exceeds a threshold,
the image is flagged as adversarially perturbed and predictions
are annotated with a warning.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO


class DualStreamDefense:
    """
    Two-stream inference pipeline combining YOLO detection with
    FFT-based adversarial patch detection.
    """

    def __init__(
        self,
        model: YOLO,
        fft_threshold: float = 0.3,
        high_freq_ratio: float = 0.25,
        conf_threshold: float = 0.25,
    ):
        """
        Args:
            model:           Trained YOLO model for Stream 1.
            fft_threshold:   Anomaly score above which an image is flagged adversarial.
            high_freq_ratio: Fraction of the frequency spectrum considered "high-frequency".
            conf_threshold:  Confidence threshold for YOLO predictions.
        """
        self.model = model
        self.fft_threshold = fft_threshold
        self.high_freq_ratio = high_freq_ratio
        self.conf_threshold = conf_threshold

    # ── Stream 1: YOLO Detection ──────────────────────────────────────────

    def yolo_predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run standard YOLO inference on a single image.
        Tuned to reduce fragmented detections (merging parts into whole).
        """
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=0.45,           # Higher IOU threshold to keep only strong boxes
            agnostic_nms=True,  # Merge overlapping boxes across different class labels
            verbose=False,
        )
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                det = {
                    "bbox": boxes.xyxy[i].cpu().numpy().tolist(),
                    "confidence": float(boxes.conf[i].cpu()),
                    "class_id": int(boxes.cls[i].cpu()),
                    "class_name": result.names[int(boxes.cls[i].cpu())],
                }
                detections.append(det)
        return detections

    # ── Stream 2: FFT Anomaly Detection ───────────────────────────────────

    def compute_anomaly_score(self, image: np.ndarray) -> float:
        """
        Compute a high-frequency anomaly score using 2D FFT.

        Adversarial patches tend to introduce sharp, high-frequency patterns
        that are unusual in natural images. This method quantifies the
        proportion of spectral energy in the high-frequency band.

        Args:
            image: BGR uint8 image.

        Returns:
            Anomaly score in [0, 1]. Higher = more likely adversarial.
        """
        # Convert to grayscale for spectral analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 2D FFT and shift zero-frequency to center
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)

        # Magnitude spectrum (log scale for numerical stability)
        magnitude = np.abs(f_shift)
        magnitude = np.log1p(magnitude)

        # Total spectral energy
        total_energy = np.sum(magnitude)
        if total_energy < 1e-10:
            return 0.0

        # Create a mask that selects the high-frequency band
        # (everything outside a centered circle of radius proportional to image size)
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        radius = int(min(h, w) * (1 - self.high_freq_ratio) / 2)

        # Low-frequency mask (circle in center)
        Y, X = np.ogrid[:h, :w]
        low_freq_mask = ((X - cx) ** 2 + (Y - cy) ** 2) <= radius ** 2

        # High-frequency energy = total - low
        low_freq_energy = np.sum(magnitude[low_freq_mask])
        high_freq_energy = total_energy - low_freq_energy

        # Anomaly score: ratio of high-freq energy to total
        anomaly_score = float(high_freq_energy / total_energy)

        return anomaly_score

    # ── Fusion Logic ──────────────────────────────────────────────────────

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Full dual-stream prediction with adversarial detection.

        Args:
            image: BGR uint8 image.

        Returns:
            Dict with keys:
              - detections: list of YOLO detection dicts
              - anomaly_score: float
              - is_adversarial: bool
              - defense_action: str describing what the defense did
        """
        # Stream 1: YOLO detection
        detections = self.yolo_predict(image)

        # Stream 2: FFT anomaly score
        anomaly_score = self.compute_anomaly_score(image)

        # Fusion decision
        is_adversarial = anomaly_score > self.fft_threshold

        if is_adversarial:
            defense_action = (
                f"⚠ ADVERSARIAL DETECTED (score={anomaly_score:.3f} > "
                f"threshold={self.fft_threshold:.3f}). "
                f"Predictions may be unreliable."
            )
        else:
            defense_action = (
                f"✓ Clean image (score={anomaly_score:.3f} ≤ "
                f"threshold={self.fft_threshold:.3f}). "
                f"Trusting YOLO predictions."
            )

        return {
            "detections": detections,
            "anomaly_score": anomaly_score,
            "is_adversarial": is_adversarial,
            "defense_action": defense_action,
        }

    # ── Batch Inference ───────────────────────────────────────────────────

    def predict_batch(
        self, images: List[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Run dual-stream prediction on a batch of images."""
        return [self.predict(img) for img in images]

    def compute_anomaly_score_region(
        self, image: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> float:
        """
        Compute anomaly score for a specific region (bounding box).

        Useful for per-object adversarial detection rather than whole-image.

        Args:
            image: BGR uint8 image.
            bbox:  (x1, y1, x2, y2) pixel coordinates.

        Returns:
            Regional anomaly score.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

        if x2 - x1 < 8 or y2 - y1 < 8:
            return 0.0  # Region too small for meaningful FFT

        region = image[y1:y2, x1:x2]
        return self.compute_anomaly_score(region)
