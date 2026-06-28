import os
import argparse
from config import Config, DefenseMode
from trainer import AdversarialTrainer
from evaluator import RobustnessEvaluator
from ultralytics import YOLO
import yaml

def run_pipeline(args):
    # 1. Initialize Config
    config = Config(
        data_yaml=args.data,
        model_weights=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        defense_mode=DefenseMode(args.defense)
    )
    
    # 2. Dataset Preparation & Training
    best_model_path = args.model
    if args.train:
        print("\n--- Starting Training Phase ---")
        # Handle resume logic
        if args.resume:
            # Look for checkpoint in multiple possible nested locations
            possible_checkpoints = [
                os.path.join(config.output_dir, "train", "adversarial_finetune", "weights", "last.pt"),
                os.path.join("runs", "detect", config.output_dir, "train", "adversarial_finetune", "weights", "last.pt"),
                "runs/detect/runs/adversarial/train/adversarial_finetune/weights/last.pt"
            ]
            checkpoint = None
            for cp in possible_checkpoints:
                if os.path.exists(cp):
                    checkpoint = os.path.abspath(cp)
                    break
            
            if checkpoint:
                print(f"[Pipeline] Resuming from checkpoint: {checkpoint}")
                config.model_weights = checkpoint
            else:
                print(f"[Pipeline] Warning: No checkpoint found in expected locations. Starting from scratch.")
        
        trainer = AdversarialTrainer(config)
        
        # Pass resume to YOLO if model_weights is a checkpoint
        if args.resume and config.model_weights.endswith("last.pt"):
             # We need to hack the trainer a bit since Ultralytics expects 'resume=True'
             # but we also need the model to be the checkpoint path
             pass 

        best_model_path = trainer.train()
        print(f"Training complete. Best model: {best_model_path}")

    # 3. Load Trained Model
    model = YOLO(best_model_path)
    
    # 4. Evaluation
    print("\n--- Starting Evaluation Phase ---")
    evaluator = RobustnessEvaluator(model, config)
    
    # Resolve validation paths from yaml
    with open(config.data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    val_images_dir = data['val']
    if not os.path.isabs(val_images_dir):
        val_images_dir = os.path.join(os.path.dirname(config.data_yaml), val_images_dir)
        
    # Infer label dir
    val_labels_dir = os.path.join(os.path.dirname(val_images_dir), 'labels')
    if not os.path.exists(val_labels_dir):
         # Try parent/labels/val
         val_labels_dir = os.path.join(os.path.dirname(os.path.dirname(val_images_dir)), 'labels', os.path.basename(val_images_dir))

    evaluator.run_full_evaluation(val_images_dir, val_labels_dir)
    
    report_path = os.path.join(config.output_dir, "evaluation_report.txt")
    evaluator.save_report(report_path)
    print(f"Full pipeline execution complete. Report at {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KAAL: Robust ML Pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained or starting weights")
    parser.add_argument("--train", action="store_true", help="Run training before evaluation")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--defense", type=str, default="dual_stream", choices=[d.value for d in DefenseMode], help="Defense strategy")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    
    args = parser.parse_args()
    run_pipeline(args)
