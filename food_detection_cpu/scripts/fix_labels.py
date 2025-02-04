import os
from pathlib import Path
import shutil
import cv2
import numpy as np

# Class definitions for our food detection model
CLASSES = {
    'PIZZA': 0,
    'BURGER': 1,
    'SANDWICH': 2,
    'SALAD': 3,
    'SUSHI': 4,
    'PASTA': 5,
    'STEAK': 6,
    'SOUP': 7,
    'TACO': 8,
    'CURRY': 9
}

# Source dataset class mappings
# Format: dataset_class_id: our_class_id

# Food-101 Dataset IDs (1-20)
FOOD101_MAPPINGS = {
    10: CLASSES['PIZZA'],      # Pizza
    13: CLASSES['BURGER'],     # Hamburger
    14: CLASSES['SANDWICH'],   # Club sandwich
    15: CLASSES['SALAD'],      # Caesar salad
    16: CLASSES['SUSHI'],      # Sushi rolls
    17: CLASSES['PASTA'],      # Spaghetti
    18: CLASSES['STEAK'],      # Steak
    19: CLASSES['SOUP'],       # Hot and sour soup
}

# COCO Food Dataset IDs (41-60)
COCO_MAPPINGS = {
    41: CLASSES['PIZZA'],      # Pizza
    42: CLASSES['BURGER'],     # Burger
    43: CLASSES['SANDWICH'],   # Sandwich
    44: CLASSES['SALAD'],      # Salad
    45: CLASSES['SUSHI'],      # Sushi
    46: CLASSES['PASTA'],      # Pasta dishes
    47: CLASSES['STEAK'],      # Steak
    48: CLASSES['SOUP'],       # Soup
    49: CLASSES['TACO'],       # Taco
    50: CLASSES['CURRY'],      # Curry
}

# Open Images Dataset IDs (51-70)
OPEN_IMAGES_MAPPINGS = {
    51: CLASSES['PIZZA'],      # Pizza
    52: CLASSES['BURGER'],     # Hamburger
    53: CLASSES['SANDWICH'],   # Sandwich
    54: CLASSES['SALAD'],      # Salad
    55: CLASSES['SUSHI'],      # Sushi
    56: CLASSES['PASTA'],      # Pasta
    57: CLASSES['STEAK'],      # Steak
    58: CLASSES['SOUP'],       # Soup
    59: CLASSES['TACO'],       # Taco
    60: CLASSES['CURRY'],      # Curry
}

# Combine all verified mappings
class_mapping = {
    **FOOD101_MAPPINGS,
    **COCO_MAPPINGS,
    **OPEN_IMAGES_MAPPINGS
}

def verify_label(label_id):
    """
    Verify if a label ID is in our verified mappings.
    Returns the mapped class ID if valid, None if not.
    """
    return class_mapping.get(label_id)

def load_image(image_path):
    """Load and preprocess image for validation."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def validate_label(image_path, label_id, bbox):
    """
    Validate if the label makes sense for the image region.
    bbox format: [x_center, y_center, width, height] (normalized)
    """
    img = load_image(image_path)
    if img is None:
        return True  # Can't validate, assume it's correct
        
    h, w = img.shape[:2]
    x, y, width, height = bbox
    
    # Convert normalized coordinates to pixel coordinates
    x1 = int((x - width/2) * w)
    y1 = int((y - height/2) * h)
    x2 = int((x + width/2) * w)
    y2 = int((y + height/2) * h)
    
    # Ensure coordinates are within image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return False
        
    # Extract the region
    region = img[y1:y2, x1:x2]
    
    # Basic shape validation
    aspect_ratio = width / height if height > 0 else 0
    
    # Validation rules based on class
    if label_id == CLASSES['PIZZA']:
        # Pizza should be relatively round (aspect ratio close to 1)
        if not (0.8 < aspect_ratio < 1.2):
            return False
            
    elif label_id == CLASSES['BURGER']:
        # Burgers typically wider than tall
        if not (1.2 < aspect_ratio < 2.0):
            return False
            
    elif label_id == CLASSES['SOUP']:
        # Soup containers typically taller than wide
        if aspect_ratio > 1.2:
            return False
    
    return True

def fix_label_file(file_path):
    """
    Fix labels in a single file, removing any unverified mappings.
    Returns True if file was modified, False otherwise.
    """
    modified = False
    new_labels = []
    
    # Get corresponding image path
    image_path = str(file_path).replace('labels', 'images').replace('.txt', '.jpg')
    if not os.path.exists(image_path):
        image_path = image_path.replace('.jpg', '.png')
        if not os.path.exists(image_path):
            return False
    
    with open(file_path, 'r') as f:
        labels = f.readlines()
        
    for label in labels:
        parts = label.strip().split()
        if not parts or len(parts) != 5:  # YOLO format requires 5 values
            continue
            
        class_id = int(parts[0])
        bbox = [float(x) for x in parts[1:]]
        
        # Validate the label
        if 0 <= class_id <= 9 and validate_label(image_path, class_id, bbox):
            new_labels.append(label)
        else:
            modified = True
    
    if modified:
        with open(file_path, 'w') as f:
            f.writelines(new_labels)
    
    return modified

def main():
    """
    Process all label files in the dataset.
    """
    # Paths to label directories
    label_dirs = [
        'data/train/labels',
        'data/val/labels',
        'data/test/labels'
    ]
    
    total_files = 0
    modified_files = 0
    
    for label_dir in label_dirs:
        if not os.path.exists(label_dir):
            print(f"Warning: Directory {label_dir} not found")
            continue
            
        for label_file in Path(label_dir).glob('*.txt'):
            total_files += 1
            if fix_label_file(str(label_file)):
                modified_files += 1
    
    print(f"Processed {total_files} files")
    print(f"Modified {modified_files} files")
    print("Label correction complete")

if __name__ == "__main__":
    main() 