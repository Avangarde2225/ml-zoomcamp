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
        example_files = ['pizza.jpg', 'burger.jpg', 'sushi.jpg']
        
        for file in example_files:
            file_path = self.examples_dir / file
            if not file_path.exists():
                # Create a blank image with text
                img = np.zeros((300, 400, 3), dtype=np.uint8)
                img.fill(255)  # White background
                
                # Add text
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, f'Sample {file.split(".")[0]} image', 
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
            
            for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                class_name = self.class_names[int(cls)]
                confidence = float(conf)
                detections.append(f"{class_name}: {confidence:.2%}")
            
            if not detections:
                return results.plot(), "No food items detected"
            
            return results.plot(), "\n".join(detections)
            
        except Exception as e:
            print(f"Error in detection: {str(e)}")
            return None, f"Error during detection: {str(e)}"

    def create_interface(self):
        """Create and configure the Gradio interface"""
        interface = gr.Interface(
            fn=self.detect_food,
            inputs=gr.Image(type="pil", label="Upload an image"),
            outputs=[
                gr.Image(type="numpy", label="Detected Food Items"),
                gr.Textbox(label="Detections")
            ],
            title="Food Detection System",
            description=self.model_description,
            examples=[
                [str(self.examples_dir / "pizza.jpg")],
                [str(self.examples_dir / "burger.jpg")],
                [str(self.examples_dir / "sushi.jpg")]
            ],
            allow_flagging="never"
        )
        return interface

    def run(self):
        """Run the Gradio app"""
        interface = self.create_interface()
        # Configure for AWS App Runner
        port = int(os.environ.get("PORT", 7860))
        interface.launch(
            server_name="0.0.0.0",  # Required for AWS
            server_port=port,        # Use PORT from environment
            share=False,
            show_error=True,         # Show detailed error messages
            enable_queue=True,       # Enable request queuing
            max_threads=40           # Increase thread limit
        )

def main():
    try:
        app = FoodDetectionApp()
        app.run()
    except Exception as e:
        print(f"Application error: {str(e)}")

if __name__ == "__main__":
    main() 