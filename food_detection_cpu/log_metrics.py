import mlflow
import json
import os
from pathlib import Path

def log_yolo_metrics():
    """Log YOLOv8 training metrics to MLflow"""
    
    # Start MLflow run
    mlflow.set_experiment("food_detection_yolov8")
    
    with mlflow.start_run(run_name="yolov8n_training"):
        # Log model parameters
        model_params = {
            "model": "yolov8n",
            "epochs": 50,
            "image_size": 640,
            "batch_size": 16,
            "device": "cpu"
        }
        mlflow.log_params(model_params)
        
        # Log metrics from the latest training
        results_file = Path("runs/detect/train9/results.json")
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
                
            # Log metrics
            metrics = {
                "mAP50": results["metrics/mAP50(B)"],
                "mAP50-95": results["metrics/mAP50-95(B)"],
                "precision": results["metrics/precision(B)"],
                "recall": results["metrics/recall(B)"]
            }
            mlflow.log_metrics(metrics)
            
            # Log the model weights as artifact
            mlflow.log_artifact("runs/detect/train9/weights/best.pt", "model")
            
            # Log example predictions
            inference_dir = Path("inference_results")
            if inference_dir.exists():
                for img in inference_dir.glob("*.png"):
                    mlflow.log_artifact(str(img), "predictions")

if __name__ == "__main__":
    log_yolo_metrics() 