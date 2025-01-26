import gradio as gr
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import warnings
import torch
import os

# Suppress future warnings from PyTorch
warnings.filterwarnings('ignore', category=FutureWarning)

class FoodDetectionApp:
    def __init__(self):
        """Initialize the Food Detection Application"""
        # Disable gradients for inference
        torch.set_grad_enabled(False)
        
        # Load model
        self.model = YOLO('runs/detect/train9/weights/best.pt')
        self.model.to('cpu')  # Ensure model is on CPU
        
        self.class_names = ['pizza', 'burger', 'sandwich', 'salad', 'pasta', 'sushi', 'steak', 'soup', 'taco', 'curry']
        
        # Model metrics - can be updated from evaluation results
        self.metrics = {
            'mAP50': 0.630,
            'mAP50-95': 0.562,
            'precision': 0.636,
            'recall': 0.566
        }
        
        # Ensure examples directory exists
        self.examples_dir = Path('examples')
        self.examples_dir.mkdir(exist_ok=True)
        
        # Create example images if they don't exist
        self.setup_example_images()
        
    def setup_example_images(self):
        """Create placeholder images if examples don't exist"""
        example_files = [
            ('pizza_detected.png', 'Pizza'),
            ('burger_detected.png', 'Burger'),
            ('sushi_detected.png', 'Sushi')
        ]
        
        # Ensure docs/images directory exists
        docs_images_dir = Path('docs/images')
        docs_images_dir.mkdir(parents=True, exist_ok=True)
        
        for file, label in example_files:
            file_path = docs_images_dir / file
            if not file_path.exists():
                # Create a blank image with text
                img = np.zeros((300, 400, 3), dtype=np.uint8)
                img.fill(255)  # White background
                
                # Add text
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, f'Sample {label} Image', 
                          (50, 150), font, 1, (0, 0, 0), 2)
                
                # Save image
                cv2.imwrite(str(file_path), img)
        
    @property
    def model_description(self):
        """Generate the model description markdown"""
        return f"""## Food Detection Model Performance
- mAP50: {self.metrics['mAP50']:.3f}
- mAP50-95: {self.metrics['mAP50-95']:.3f}
- Precision: {self.metrics['precision']:.3f}
- Recall: {self.metrics['recall']:.3f}
"""

    def process_image(self, image):
        """Convert and prepare image for inference"""
        if image is None:
            return None
        if isinstance(image, str):
            # If image is a file path
            if not os.path.exists(image):
                return None
            image = gr.processing_utils.decode_base64_to_image(image)
        return np.array(image)

    def detect_food(self, image):
        """Main detection function"""
        try:
            processed_image = self.process_image(image)
            if processed_image is None:
                return None, "Error: Could not process image"
            
            results = self.model.predict(processed_image, conf=0.25, verbose=False)[0]
            detections = []
            
            # Sort detections by confidence
            detection_info = []
            for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                class_name = self.class_names[int(cls)]
                confidence = float(conf)
                detection_info.append((class_name, confidence))
            
            # Sort by confidence
            detection_info.sort(key=lambda x: x[1], reverse=True)
            
            if not detection_info:
                return results.plot(), "No food items detected in the image.\n\nTry uploading a different image with one of these food items:\n" + "\n".join([f"- {cls.title()}" for cls in self.class_names])
            
            # Format detections with emojis and confidence
            food_emojis = {
                'pizza': '🍕', 'burger': '🍔', 'sandwich': '🥪', 
                'salad': '🥗', 'pasta': '🍝', 'sushi': '🍱',
                'steak': '🥩', 'soup': '🥣', 'taco': '🌮', 'curry': '🍛'
            }
            
            detections.append("Detected Food Items:")
            for class_name, confidence in detection_info:
                emoji = food_emojis.get(class_name, '')
                detections.append(f"{emoji} {class_name.title()}: {confidence:.1%}")
            
            detections.append("\nModel Performance:")
            detections.append(f"mAP50: {self.metrics['mAP50']:.1%}")
            detections.append(f"Precision: {self.metrics['precision']:.1%}")
            detections.append(f"Recall: {self.metrics['recall']:.1%}")
            
            return results.plot(), "\n".join(detections)
            
        except Exception as e:
            print(f"Error in detection: {str(e)}")
            return None, f"Error during detection: {str(e)}\n\nPlease try again with a different image."

    def create_interface(self):
        """Create and configure the Gradio interface"""
        interface = gr.Interface(
            fn=self.detect_food,
            inputs=gr.Image(type="pil", label="Upload a food image"),
            outputs=[
                gr.Image(type="numpy", label="Detected Food Items"),
                gr.Textbox(label="Detected Objects with Confidence Scores")
            ],
            title="Food Detection System",
            description=self.model_description + "\n\n### Supported Food Classes:\n" + 
                       ", ".join([f"- {cls.title()}" for cls in self.class_names]) +
                       "\n\n### Instructions:\n" +
                       "1. Upload an image or use one of the examples below\n" +
                       "2. The model will detect food items and show bounding boxes\n" +
                       "3. Confidence scores will be displayed for each detection",
            examples=[
                [str(Path("docs/images/pizza_detected.png"))],
                [str(Path("docs/images/burger_detected.png"))],
                [str(Path("docs/images/sushi_detected.png"))]
            ],
            allow_flagging="never",
            theme="default"
        )
        return interface

    def run(self):
        """Run the Gradio app"""
        interface = self.create_interface()
        
        # Configure for AWS App Runner
        port = int(os.environ.get("PORT", 7860))
        print(f"Starting server on port {port}...")
        
        try:
            # Launch with production configuration
            interface.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=False,
                show_error=True,
                quiet=False,  # Enable all logging for debugging
                auth=None,
                prevent_thread_lock=True
            )
            print(f"Server started successfully on port {port}")
        except Exception as e:
            print(f"Failed to start server: {str(e)}")
            raise

def main():
    try:
        app = FoodDetectionApp()
        app.run()
    except Exception as e:
        print(f"Application error: {str(e)}")

if __name__ == "__main__":
    main() 