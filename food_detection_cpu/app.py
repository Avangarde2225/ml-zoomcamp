import gradio as gr
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import warnings
import torch
import os
import logging
import sys

# Suppress future warnings from PyTorch
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

class FoodDetectionApp:
    def __init__(self):
        """Initialize the Food Detection Application"""
        try:
            logging.info("Initializing Food Detection App...")
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
            
            logging.info("Food Detection App initialized successfully")
        except Exception as e:
            logging.error(f"Error during initialization: {str(e)}")
            raise
        
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
        try:
            port = int(os.environ.get("PORT", 7860))
            server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
            logging.info(f"Starting application on {server_name}:{port}")
            
            self.interface = self.create_interface()
            
            # Log environment variables for debugging
            logging.info(f"Environment variables:")
            logging.info(f"PORT: {os.environ.get('PORT')}")
            logging.info(f"GRADIO_SERVER_NAME: {os.environ.get('GRADIO_SERVER_NAME')}")
            logging.info(f"GRADIO_SERVER_PORT: {os.environ.get('GRADIO_SERVER_PORT')}")
            
            self.interface.launch(
                server_name=server_name,
                server_port=port,
                show_error=True,
                quiet=False
            )
            logging.info("Application started successfully")
        except Exception as e:
            logging.error(f"Error starting application: {str(e)}")
            sys.exit(1)  # Exit with error code

def main():
    try:
        logging.info("Starting Food Detection Application")
        app = FoodDetectionApp()
        app.run()
    except Exception as e:
        logging.error(f"Application failed to start: {str(e)}")
        sys.exit(1)  # Exit with error code to signal failure

if __name__ == "__main__":
    main() 