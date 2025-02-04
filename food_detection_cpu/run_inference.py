from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import numpy as np

def validate_prediction(image, class_id, box, conf):
    """Additional validation based on image properties"""
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    roi = image[y1:y2, x1:x2]
    
    if roi.size == 0:
        return False
        
    # Calculate color statistics
    avg_color = np.mean(roi, axis=(0,1))
    std_color = np.std(roi, axis=(0,1))
    
    # Validate based on rules
    if class_id == 1:  # burger
        # Burgers should have significant color variation (bun, patty, etc.)
        return std_color.mean() > 30
    elif class_id == 7:  # soup
        # Soup should have more uniform color and be in a container
        aspect_ratio = (x2-x1)/(y2-y1)
        return 0.8 < aspect_ratio < 1.5 and std_color.mean() < 50
    
    return True

def run_inference():
    # Load model
    model = YOLO("runs/detect/train9/weights/best.pt")
    
    # Create output directory
    os.makedirs("inference_results", exist_ok=True)
    
    # Define test images
    test_images = [
        "data/test/images/burger_01.jpg",
        "data/test/images/sandwich_01.jpg",
        "data/test/images/pizza_01.jpg"
    ]
    
    # Class mapping
    class_names = {
        0: 'pizza',
        1: 'burger',
        2: 'sandwich',
        3: 'salad',
        4: 'sushi',
        5: 'pasta',
        6: 'steak',
        7: 'soup',
        8: 'taco',
        9: 'curry'
    }
    
    # Run inference with much higher confidence threshold
    for img_path in test_images:
        if os.path.exists(img_path):
            # Load image for validation
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load image: {img_path}")
                continue
                
            # Run prediction with higher confidence threshold
            results = model.predict(img_path, conf=0.85)[0]  # Increased confidence threshold significantly
            
            # Get base name for output file
            base_name = Path(img_path).name.split('.')[0]
            
            # Save the prediction
            output_path = os.path.join("inference_results", f"{base_name}_detected.png")
            
            # Filter and validate detections
            valid_detections = []
            for box in results.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Additional validation
                if validate_prediction(image, class_id, box, conf):
                    valid_detections.append((class_id, conf, box))
                    class_name = class_names.get(class_id, f"unknown_{class_id}")
                    print(f"\nProcessing: {img_path}")
                    print(f"Detected {class_name} with confidence {conf:.2f}")
            
            # Draw only valid detections
            if valid_detections:
                result_image = image.copy()
                for class_id, conf, box in valid_detections:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = class_names.get(class_id, f"unknown_{class_id}")
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(result_image, f"{class_name} {conf:.2f}", (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imwrite(output_path, result_image)
                print(f"Saved prediction to: {output_path}")
            else:
                print(f"No valid detections for {img_path}")
        else:
            print(f"Image not found: {img_path}")

if __name__ == "__main__":
    run_inference() 