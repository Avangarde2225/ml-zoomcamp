from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time

class FoodDetector:
    def __init__(self, model_path='runs/detect/train9/weights/best.pt'):
        self.model = YOLO(model_path)
        self.names = ['pizza', 'burger', 'sandwich', 'salad', 'sushi', 'pasta', 'steak', 'soup', 'taco', 'curry']
        
    def predict_single_image(self, image_path, conf_threshold=0.25, save_plot=True):
        """
        Run inference on a single image
        """
        print(f"\nProcessing image: {image_path}")
        start_time = time.time()
        
        # Run inference
        results = self.model.predict(
            source=str(image_path),
            conf=conf_threshold,
            save=False
        )[0]
        
        # Process results
        detections = []
        if len(results.boxes) > 0:
            for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                class_name = self.names[int(cls)]
                detections.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'box': box.tolist()
                })
        
        # Save visualization if requested
        if save_plot and len(detections) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(results.plot())
            plt.title(f'Detections for {Path(image_path).name}')
            plt.axis('off')
            save_path = f'inference_results/{Path(image_path).stem}_detected.png'
            Path('inference_results').mkdir(exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        
        inference_time = time.time() - start_time
        
        return {
            'file_name': Path(image_path).name,
            'inference_time': inference_time,
            'num_detections': len(detections),
            'detections': detections
        }

    def batch_inference(self, image_dir, conf_threshold=0.25, save_plots=True):
        """
        Run inference on a directory of images
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise ValueError(f"Directory {image_dir} does not exist")
        
        # Get all image files
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.jpeg')) + list(image_dir.glob('*.png'))
        if not image_files:
            raise ValueError(f"No image files found in {image_dir}")
        
        print(f"Found {len(image_files)} images. Starting batch inference...")
        
        # Process each image
        results = []
        for img_path in image_files:
            result = self.predict_single_image(img_path, conf_threshold, save_plots)
            results.append(result)
            
            # Print detection summary
            print(f"\nResults for {result['file_name']}:")
            print(f"- Inference time: {result['inference_time']:.3f} seconds")
            print(f"- Number of detections: {result['num_detections']}")
            for det in result['detections']:
                print(f"  * {det['class']}: {det['confidence']:.2%}")
        
        return results

def main():
    # Initialize detector
    detector = FoodDetector()
    
    # Example usage
    print("Food Detection Inference Tool")
    print("1. Single image inference")
    print("2. Batch inference")
    
    choice = input("\nSelect mode (1 or 2): ")
    
    if choice == '1':
        image_path = input("Enter image path: ")
        result = detector.predict_single_image(image_path)
        
        print("\nDetection Results:")
        print(f"Inference time: {result['inference_time']:.3f} seconds")
        print(f"Number of detections: {result['num_detections']}")
        for det in result['detections']:
            print(f"- {det['class']}: {det['confidence']:.2%}")
            
    elif choice == '2':
        image_dir = input("Enter directory path containing images: ")
        results = detector.batch_inference(image_dir)
        
        print("\nBatch Processing Summary:")
        print(f"Total images processed: {len(results)}")
        total_detections = sum(r['num_detections'] for r in results)
        print(f"Total detections: {total_detections}")
        avg_time = np.mean([r['inference_time'] for r in results])
        print(f"Average inference time: {avg_time:.3f} seconds")
    
    else:
        print("Invalid choice. Please run again and select 1 or 2.")

if __name__ == "__main__":
    main() 