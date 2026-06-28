import os
import yaml
import cv2
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def validate_yolo_dataset(data_yaml_path):
    print(f"--- Validating Dataset: {data_yaml_path} ---")
    
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    base_dir = Path(data_yaml_path).parent
    splits = ['train', 'val', 'test']
    
    for split in splits:
        if split not in data:
            print(f"[!] Split '{split}' not found in yaml.")
            continue
            
        img_path = Path(data[split])
        if not img_path.is_absolute():
            img_path = base_dir / img_path
            
        if not img_path.exists():
            print(f"[!] {split} image directory not found: {img_path}")
            continue
            
        # Infer labels path
        label_path = img_path.parent / 'labels'
        if not label_path.exists():
            # Try sibling
            label_path = img_path.parent.parent / 'labels' / img_path.name
            
        print(f"\nChecking {split} split:")
        print(f"  Images: {img_path}")
        print(f"  Labels: {label_path}")
        
        img_files = list(img_path.glob('*.*'))
        img_files = [f for f in img_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        if not img_files:
            print(f"  [!] No images found in {img_path}")
            continue
            
        stats = {
            'total_images': len(img_files),
            'missing_labels': 0,
            'empty_labels': 0,
            'corrupt_images': 0,
            'class_counts': defaultdict(int)
        }
        
        for img_f in tqdm(img_files, desc=f"Validating {split}"):
            # Check if image is corrupt
            img = cv2.imread(str(img_f))
            if img is None:
                stats['corrupt_images'] += 1
                continue
                
            h, w = img.shape[:2]
            
            label_f = label_path / (img_f.stem + ".txt")
            if not label_f.exists():
                stats['missing_labels'] += 1
                continue
                
            with open(label_f, 'r') as lf:
                lines = lf.readlines()
                if not lines:
                    stats['empty_labels'] += 1
                    continue
                    
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        stats['class_counts'][cls_id] += 1
                        
                        # Validate coordinates
                        try:
                            cx, cy, bw, bh = map(float, parts[1:5])
                            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= bw <= 1 and 0 <= bh <= 1):
                                print(f"  [!] Invalid coordinates in {label_f}: {line.strip()}")
                        except ValueError:
                            print(f"  [!] Malformed label in {label_f}: {line.strip()}")

        print(f"  Results for {split}:")
        print(f"    Total Images:    {stats['total_images']}")
        print(f"    Corrupt Images:  {stats['corrupt_images']}")
        print(f"    Missing Labels:  {stats['missing_labels']}")
        print(f"    Empty Labels:    {stats['empty_labels']}")
        print(f"    Class Distribution:")
        for cls_id, count in sorted(stats['class_counts'].items()):
            name = data['names'][cls_id] if 'names' in data and cls_id < len(data['names']) else f"ID {cls_id}"
            print(f"      - {name}: {count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    args = parser.parse_args()
    validate_yolo_dataset(args.data)
