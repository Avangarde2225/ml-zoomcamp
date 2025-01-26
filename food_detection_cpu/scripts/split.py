import os
import shutil
import random
from pathlib import Path

def split_data(source_dir: Path, train_dir: Path, val_dir: Path, test_dir: Path,
               train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Split data into train, validation and test sets."""
    
    # Create destination directories if they don't exist
    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(source_dir.glob("*.jpg"))
    random.shuffle(image_files)
    
    # Calculate split indices
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split files
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    # Copy files to respective directories
    for img_path in train_files:
        shutil.copy2(img_path, train_dir / img_path.name)
        
    for img_path in val_files:
        shutil.copy2(img_path, val_dir / img_path.name)
        
    for img_path in test_files:
        shutil.copy2(img_path, test_dir / img_path.name)
    
    print(f"Total images: {n_total}")
    print(f"Training: {len(train_files)} images")
    print(f"Validation: {len(val_files)} images")
    print(f"Test: {len(test_files)} images")

def main():
    # Define paths - using absolute paths from food_detection_cpu root
    source_dir = Path("data/raw")  # where your original images are
    base_dir = Path("data")
    
    train_dir = base_dir / "train" / "images"
    val_dir = base_dir / "val" / "images"
    test_dir = base_dir / "test" / "images"
    
    # Create label directories too
    train_label_dir = base_dir / "train" / "labels"
    val_label_dir = base_dir / "val" / "labels"
    test_label_dir = base_dir / "test" / "labels"
    
    # Create all directories
    for d in [train_dir, val_dir, test_dir, train_label_dir, val_label_dir, test_label_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Split the data
    split_data(source_dir, train_dir, val_dir, test_dir)

if __name__ == "__main__":
    main() 