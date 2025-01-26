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
    def __init__(self, model_path='runs/detect/train9/weights/best.pt'):
        """Initialize the Food Detection Application"""
        # Disable gradients for inference
        torch.set_grad_enabled(False)
        
        # Load model
        self.model = YOLO(model_path)
        self.model.to('cpu')  # Ensure model is on CPU
        
        self.names = ['pizza', 'burger', 'sandwich', 'salad', 'sushi', 
                     'pasta', 'steak', 'soup', 'taco', 'curry']
        
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
        return f"""
        ## Model Performance
        - mAP50: {self.metrics['mAP50']:.3f}
        - mAP50-95: {self.metrics['mAP50-95']:.3f}
        - Precision: {self.metrics['precision']:.3f}
        - Recall: {self.metrics['recall']:.3f}

        ## Supported Food Types
        This model can detect:
        - Pizza 🍕
        - Burger 🍔
        - Sandwich 🥪
        - Salad 🥗
        - Sushi 🍱
        - Pasta 🍝
        - Steak 🥩
        - Soup 🥣
        - Taco 🌮
        - Curry 🍛

        ## Image Requirements
        - Format: JPG, PNG
        - Size: Any (will be automatically resized)
        - Content: Clear food photos
        """

    def process_image(self, image):
        """Convert and prepare image for inference"""
        try:
            if image is None:
                raise ValueError("No image provided")
                
            if isinstance(image, str):
                image = Image.open(image)
                
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            return image
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None

    def detect_food(self, image):
        """Main detection function"""
        try:
            if image is None:
                return None, "No image provided"

            # Preprocess image
            processed_image = self.process_image(image)
            if processed_image is None:
                return None, "Error processing image"
            
            # Run inference with lower confidence threshold
            results = self.model.predict(processed_image, conf=0.25, verbose=False)
            result = results[0]
            
            # Print debug information
            print(f"Number of detections: {len(result.boxes)}")
            print(f"Confidence scores: {result.boxes.conf if len(result.boxes) > 0 else 'No detections'}")
            
            # Generate visualization
            result_plot = result.plot()
            result_plot = cv2.cvtColor(result_plot, cv2.COLOR_BGR2RGB)
            
            # Generate detection descriptions
            descriptions = []
            if len(result.boxes) > 0:  # Check if there are any detections
                for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    food_type = self.names[int(cls)]
                    confidence = float(conf)
                    descriptions.append(f"{food_type}: {confidence:.2%} confidence")
                    print(f"Detected {food_type} with confidence {confidence:.2%}")
            
            description_text = "\n".join(descriptions) if descriptions else "No food items detected"
            
            return result_plot, description_text
            
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            return None, f"Error during detection: {str(e)}"

    def create_interface(self):
        """Create and configure the Gradio interface"""
        try:
            interface = gr.Interface(
                fn=self.detect_food,
                inputs=gr.Image(type="pil", label="Upload Food Image"),
                outputs=[
                    gr.Image(type="numpy", label="Detected Food Items"),
                    gr.Textbox(label="Detections")
                ],
                title="Food Detection AI",
                description=self.model_description,
                examples=[
                    [str(self.examples_dir / "pizza.jpg")],
                    [str(self.examples_dir / "burger.jpg")],
                    [str(self.examples_dir / "sushi.jpg")]
                ]
            )
            return interface
        except Exception as e:
            print(f"Error creating interface: {str(e)}")
            raise

    def run(self, server_name="0.0.0.0", server_port=7860):
        """Run the Gradio app"""
        try:
            interface = self.create_interface()
            interface.launch(
                server_name=server_name,
                server_port=server_port,
                share=False
            )
        except Exception as e:
            print(f"Error launching app: {str(e)}")
            raise

def main():
    try:
        app = FoodDetectionApp()
        app.run()
    except Exception as e:
        print(f"Application error: {str(e)}")

if __name__ == "__main__":
    main() 