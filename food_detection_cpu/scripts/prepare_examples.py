from pathlib import Path
import shutil
import random

def prepare_examples():
    """Copy a few good examples from test set to examples directory."""
    # Create examples directory
    examples_dir = Path('examples')
    examples_dir.mkdir(exist_ok=True)
    
    # Source directories
    test_dir = Path('data/test/images')
    
    # Select one example for each class if available
    class_names = ['pizza', 'burger', 'sandwich', 'salad', 'sushi', 
                  'pasta', 'steak', 'soup', 'taco', 'curry']
    
    # Get all test images
    test_images = list(test_dir.glob('*.jpg'))
    
    # Randomly select 3 images
    if test_images:
        selected = random.sample(test_images, min(3, len(test_images)))
        for i, img_path in enumerate(selected):
            target_path = examples_dir / f"example_{i+1}.jpg"
            shutil.copy2(img_path, target_path)
            print(f"Copied {img_path.name} to {target_path}")
    else:
        print("No test images found!")

if __name__ == "__main__":
    prepare_examples() 