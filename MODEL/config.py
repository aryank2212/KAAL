"""
config.py — Centralized configuration for the YOLOv8 adversarial robustness pipeline.

All hyperparameters, paths, thresholds, and defense mode settings are managed here
so that nothing is hardcoded across the codebase.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Optional


class DefenseMode(Enum):
    """Available defense strategies during training and inference."""
    NONE = "none"                              # Baseline: no adversarial defense
    ADVERSARIAL_TRAINING = "adversarial_training"  # Train on clean + adversarial, standard inference
    DUAL_STREAM = "dual_stream"                # Adversarial training + FFT dual-stream at inference


class PatchType(Enum):
    """Type of adversarial patch to generate."""
    NOISE = "noise"        # Random pixel noise
    PATTERN = "pattern"    # Structured high-frequency pattern
    LOADED = "loaded"      # Load from an image file


@dataclass
class Config:
    """
    Master configuration object.

    Populate from CLI args (main.py) or instantiate directly for programmatic use.
    """

    # ── Paths ──────────────────────────────────────────────────────────────
    data_yaml: str = "data.yaml"               # Path to YOLO-format data.yaml
    model_weights: str = "yolov8m.pt"          # Pretrained model checkpoint
    output_dir: str = "runs/adversarial"       # Directory for all outputs
    patch_image_path: Optional[str] = None     # Path to a pre-made patch (PatchType.LOADED)

    # ── Training ───────────────────────────────────────────────────────────
    epochs: int = 100
    batch_size: int = 8
    lr: float = 0.001
    imgsz: int = 640
    device: str = "0"                          # "0" for GPU, "cpu" for CPU fallback

    # ── Adversarial Patch ──────────────────────────────────────────────────
    patch_prob: float = 0.5                    # Probability of injecting a patch per image
    patch_scale_range: Tuple[float, float] = (0.1, 0.3)  # Patch size as fraction of bbox
    patch_type: PatchType = PatchType.NOISE    # Which patch generation method to use
    patch_alpha: float = 0.8                   # Blending alpha (1.0 = fully opaque)

    # ── Defense / Dual-Stream ──────────────────────────────────────────────
    defense_mode: DefenseMode = DefenseMode.DUAL_STREAM
    fft_threshold: float = 0.3                 # High-frequency anomaly threshold
    fft_high_freq_ratio: float = 0.25          # Fraction of spectrum considered "high-freq"

    # ── Evaluation ─────────────────────────────────────────────────────────
    conf_threshold: float = 0.45               # Confidence threshold for predictions (increased to avoid weak edge detection)
    iou_threshold: float = 0.40                # IoU threshold for mAP/NMS (lowered to force highly overlapping fragments to merge)

    # ── Visualization ──────────────────────────────────────────────────────
    num_vis_samples: int = 8                   # Number of sample images to visualize

    def __post_init__(self):
        """Validate configuration values after initialization."""
        assert 0.0 <= self.patch_prob <= 1.0, "patch_prob must be in [0, 1]"
        assert 0.0 < self.patch_alpha <= 1.0, "patch_alpha must be in (0, 1]"
        assert 0.0 < self.fft_threshold < 1.0, "fft_threshold must be in (0, 1)"
        assert self.patch_scale_range[0] < self.patch_scale_range[1], \
            "patch_scale_range[0] must be < patch_scale_range[1]"
        assert self.epochs > 0, "epochs must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
