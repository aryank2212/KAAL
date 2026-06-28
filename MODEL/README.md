# YOLOv8 Adversarial Robustness Pipeline

A production-quality pipeline for fine-tuning YOLOv8 with adversarial patch robustness and dual-stream defense.

## Features

- **Adversarial patch injection** — noise, structured patterns, or custom patch images
- **Mixed training** — fine-tune on both clean and adversarial-augmented images
- **Dual-stream defense** — FFT-based anomaly detection fused with YOLO predictions
- **Three defense modes** — switch between no defense, adversarial training only, or full dual-stream
- **Comprehensive evaluation** — mAP on clean, attacked, and defended scenarios
- **Visualizations** — before/after attacks, annotated predictions, confusion matrices

---

## Project Structure

```
MODEL/
├── config.py                  # Centralized configuration (hyperparams, paths, thresholds)
├── adversarial_patch.py       # Patch generation, placement, and probabilistic injection
├── dataset.py                 # YOLO-format dataset loading + augmentation pipeline
├── trainer.py                 # Fine-tuning: clean + adversarial mixed training
├── dual_stream_defense.py     # FFT anomaly detection + fusion with YOLO predictions
├── evaluator.py               # mAP computation, confusion matrix, metrics reporting
├── visualizer.py              # Before/after visualization, bbox overlays, adversarial flags
├── main.py                    # CLI entry point (train → evaluate → visualize)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Your dataset must be in **YOLO format** with a `data.yaml` file:

```
my_dataset/
├── data.yaml
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── ...
│   └── labels/
│       ├── img001.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── test/       (optional)
    ├── images/
    └── labels/
```

Example `data.yaml`:

```yaml
train: train/images
val: val/images
nc: 3
names: ['car', 'person', 'bicycle']
```

---

## Usage

### Full Pipeline (Train → Evaluate → Visualize)

```bash
python main.py --data path/to/data.yaml --epochs 50 --batch 16 --lr 0.001 --defense dual_stream
```

### Defense Modes

| Mode | Training | Inference | Description |
|------|----------|-----------|-------------|
| `none` | Clean data only | Standard YOLO | Baseline, no defense |
| `adversarial_training` | Clean + adversarial | Standard YOLO | Robust training only |
| `dual_stream` | Clean + adversarial | YOLO + FFT anomaly | Full defense (recommended) |

```bash
# No defense (baseline)
python main.py --data data.yaml --defense none

# Adversarial training only
python main.py --data data.yaml --defense adversarial_training

# Full dual-stream defense
python main.py --data data.yaml --defense dual_stream --fft-threshold 0.3
```

### Evaluate Only (Skip Training)

```bash
python main.py --data data.yaml --eval-only --weights path/to/best.pt
```

### Visualize Only

```bash
python main.py --data data.yaml --vis-only --weights path/to/best.pt
```

### All CLI Options

```
--data              Path to data.yaml (default: data.yaml)
--weights           Pretrained model weights (default: yolov8n.pt)
--output            Output directory (default: runs/adversarial)
--epochs            Training epochs (default: 50)
--batch             Batch size (default: 16)
--lr                Learning rate (default: 0.001)
--imgsz             Image size (default: 640)
--device            Device: '0' for GPU, 'cpu' for CPU (default: 0)
--defense           Defense mode: none/adversarial_training/dual_stream
--patch-prob        Probability of patch injection (default: 0.5)
--patch-type        Patch type: noise/pattern/loaded (default: noise)
--patch-alpha       Patch blending opacity (default: 0.8)
--patch-image       Path to pre-made adversarial patch image
--fft-threshold     FFT anomaly threshold (default: 0.3)
--conf              Confidence threshold (default: 0.25)
--iou               IoU threshold for mAP (default: 0.5)
--eval-only         Skip training, evaluate only
--vis-only          Skip training + evaluation, visualize only
--skip-eval         Skip evaluation after training
--skip-vis          Skip visualizations
--num-vis           Number of visualization samples (default: 8)
```

---

## How It Works

### 1. Adversarial Patch Injection

During training, adversarial patches are injected onto objects within their bounding boxes:
- **Noise patches**: random pixel noise acting as universal perturbation
- **Pattern patches**: high-frequency checkerboard patterns
- **Loaded patches**: custom adversarial images (e.g., printed stickers)

Patches are alpha-blended within bounding boxes so annotations remain valid.

### 2. Dual-Stream Defense

At inference time, each image is processed through two streams:

- **Stream 1 (YOLO)**: Standard object detection
- **Stream 2 (FFT)**: Convert to frequency domain → measure high-frequency energy ratio

**Fusion**: If the FFT anomaly score exceeds the threshold, the image is flagged as adversarial and predictions are marked unreliable.

### 3. Evaluation

Three-pass evaluation on the validation set:
1. **Clean**: mAP on unmodified images
2. **Adversarial**: mAP on attacked images (no defense)
3. **Defended**: mAP on attacked images with dual-stream defense active

---

## Output Structure

```
runs/adversarial/
├── augmented_dataset/         # Mixed clean + adversarial training data
│   ├── train/
│   ├── val/
│   └── data.yaml
├── train/                     # Ultralytics training outputs
│   └── adversarial_finetune/
│       └── weights/
│           ├── best.pt
│           └── last.pt
├── visualizations/            # Generated images
│   ├── adversarial_comparison.png
│   ├── sample_predictions_clean.png
│   ├── sample_predictions_attacked.png
│   └── confusion_matrix.png
└── evaluation_report.txt      # Metrics summary
```

---

## Example Output

```
============================================================
  ROBUSTNESS EVALUATION REPORT
============================================================

  Scenario                   | mAP@0.5 | TP  | FP | FN
  ---------------------------+---------+-----+----+---
  Clean (baseline)           | 0.7823  | 245 | 32 | 18
  Adversarial (no defense)   | 0.4156  | 128 | 89 | 135
  Adversarial + Defense      | 0.6891  | 210 | 45 | 53

  Adversarial images flagged: 42/50
  Mean anomaly score: 0.4521
  Max anomaly score: 0.7834

  mAP drop from attack: -0.3667
  mAP recovery with defense: +0.2735
```
