from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pandas as pd

def evaluate_model():
    # Load the best model from train9 directory
    model = YOLO('runs/detect/train9/weights/best.pt')

    # Class names
    names = ['pizza', 'burger', 'sandwich', 'salad', 'sushi', 'pasta', 'steak', 'soup', 'taco', 'curry']

    # Run validation
    print("Running validation...")
    results = model.val(data='data/data.yaml')

    print("\nModel Performance:")
    print(f"mAP50: {results.results_dict['metrics/mAP50(B)']:.3f}")
    print(f"mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.3f}")
    print(f"Precision: {results.results_dict['metrics/precision(B)']:.3f}")
    print(f"Recall: {results.results_dict['metrics/recall(B)']:.3f}")

    # Extract per-class metrics
    class_metrics = pd.DataFrame({
        'Class': names,
        'Precision': results.results_dict['metrics/precision(B)'],
        'Recall': results.results_dict['metrics/recall(B)'],
        'mAP50': results.results_dict['metrics/mAP50(B)'],
        'mAP50-95': results.results_dict['metrics/mAP50-95(B)']
    })

    print("\nPer-class Performance:")
    print(class_metrics.round(3))

    # Visualize per-class metrics
    plt.figure(figsize=(12, 6))
    sns.barplot(data=class_metrics.melt(id_vars=['Class'], 
                                    value_vars=['Precision', 'Recall', 'mAP50']),
                x='Class', y='value', hue='variable')
    plt.xticks(rotation=45)
    plt.title('Per-class Performance Metrics')
    plt.tight_layout()
    plt.savefig('performance_metrics.png')
    plt.close()

    # Test on example images
    test_dir = Path('examples')
    if not test_dir.exists():
        print("\nNo example images found in 'examples' directory")
        return

    test_images = list(test_dir.glob('*.jpg'))
    for img_path in test_images:
        results = model.predict(str(img_path))
        
        for r in results:
            # Plot with boxes
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(r.plot())
            plt.title(f'Predictions for {img_path.name}')
            plt.axis('off')
            plt.savefig(f'prediction_{img_path.stem}.png')
            plt.close()
            
            # Print detections with confidence
            print(f"\nDetections in {img_path.name}:")
            for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                print(f"- {names[int(cls)]}: {conf:.2%} confidence")

if __name__ == "__main__":
    evaluate_model() 