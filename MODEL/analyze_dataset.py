import os
import glob

def analyze_bbox_sizes(dataset_path: str, split: str = "train", threshold: float = 0.05):
    """
    Analyzes YOLO format annotations to detect potentially fragmented bounding boxes
    (e.g., wings/edges annotated separately from the main body).

    Args:
        dataset_path: Root path to the YOLO dataset.
        split: "train" or "valid".
        threshold: Fractional area below which a bbox is considered "fragmented".
                   E.g., 0.05 means the box occupies less than 5% of the total image area.
    """
    labels_dir = os.path.join(dataset_path, split, "labels")
    if not os.path.exists(labels_dir):
        print(f"Error: {labels_dir} does not exist.")
        return

    txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    if not txt_files:
        print(f"No label files found in {labels_dir}.")
        return

    total_boxes = 0
    small_boxes = 0

    print(f"Analyzing {len(txt_files)} label files in {split} set for bounding box fragmentation...")

    for txt_file in txt_files:
        with open(txt_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    total_boxes += 1
                    # YOLO format: class_id x_center y_center width_normalized height_normalized
                    w = float(parts[3])
                    h = float(parts[4])
                    area = w * h

                    if area < threshold:
                        small_boxes += 1
                        print(f"[Warning] Fragmented box detected in {os.path.basename(txt_file)}: w={w:.3f}, h={h:.3f}, area={area:.4f}")

    if total_boxes == 0:
        print("No bounding boxes found.")
        return

    print("\n--- Summary ---")
    print(f"Total Bounding Boxes: {total_boxes}")
    print(f"Fragmented Boxes (Area < {threshold*100}%): {small_boxes}")
    print(f"Fragmentation Ratio: {(small_boxes/total_boxes)*100:.2f}%")
    print("If the ratio is high, consider merging bounding boxes or cleaning the dataset.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Find fragmented bounding boxes in dataset.")
    parser.add_argument("--data", type=str, default="Datasets/aircraft.v1i.yolov8", help="Path to YOLO dataset")
    parser.add_argument("--threshold", type=float, default=0.03, help="Area threshold to consider fragmented")
    args = parser.parse_args()

    analyze_bbox_sizes(args.data, "train", args.threshold)
