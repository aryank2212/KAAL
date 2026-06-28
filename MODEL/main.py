import argparse
import sys
import os

from training.adv_train import RobustDetectionTrainer
from evaluation.evaluate import run_evaluation
from visualization.compare import generate_visualizations

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Adversarial Robustness Framework")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Run adversarial training")
    train_parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights")
    train_parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch", type=int, default=16)
    train_parser.add_argument("--workers", type=int, default=0, help="Dataloader workers (set to 0 for WinError 1455)")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate robustness metrics")
    eval_parser.add_argument("--model", type=str, required=True, help="Trained model weights")
    eval_parser.add_argument("--data", type=str, required=True, help="Path to original data.yaml")
    
    # Visualize command
    vis_parser = subparsers.add_parser("visualize", help="Generate side-by-side comparison images")
    vis_parser.add_argument("--model", type=str, required=True, help="Trained model weights")
    vis_parser.add_argument("--data", type=str, required=True, help="Path to original data.yaml")
    
    args = parser.parse_args()
    
    if args.command == "train":
        print("Starting Adversarial Training...")
        trainer = RobustDetectionTrainer(overrides={
            "model": args.model,
            "data": args.data,
            "epochs": args.epochs,
            "batch": args.batch,
            "workers": args.workers,
            "project": "runs/robustness",
            "name": "adv_training_run"
        })
        trainer.train()
        print("Training completed.")
        
    elif args.command == "evaluate":
        print("Starting Robustness Evaluation Pipeline...")
        run_evaluation(args.model, args.data)
        print("Evaluation completed. Check terminal output for the Robustness Report.")
        
    elif args.command == "visualize":
        print("Starting Visualization Generation...")
        generate_visualizations(args.model, args.data)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
