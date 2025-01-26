import os
import shutil
from pathlib import Path

def copy_labels(image_dir: Path, label_dir: Path, source_label_dir: Path):
    """Copy existing labels for images from source directory."""
    # Create label directory if it doesn't exist
    label_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(image_dir.glob("*.jpg"))
    copied = 0
    
    # Copy corresponding label files
    for img_path in image_files:
        source_label = source_label_dir / f"{img_path.stem}.txt"
        target_label = label_dir / f"{img_path.stem}.txt"
        
        if source_label.exists():
            shutil.copy2(source_label, target_label)
            copied += 1
        else:
            print(f"Warning: No label found for {img_path.name}")
    
    return copied, len(image_files)

def main():
    # Define paths
    data_dir = Path("data")  # Path to your data directory
    source_label_dir = Path("runs/detect/predict/labels")  # Directory containing your existing labels
    
    total_copied = 0
    total_images = 0
    
    # Process each split
    for x in ['train', 'val', 'test']:
        image_dir = data_dir / x / "images"  # Images are in data/train/images etc.
        label_dir = data_dir / x / "labels"  # Labels will go in data/train/labels etc.
        copied, images = copy_labels(image_dir, label_dir, source_label_dir)
        print(f"Split {x}: Copied {copied} labels for {images} images")
        total_copied += copied
        total_images += images
    
    print(f"\nTotal: {total_copied} labels for {total_images} images")

if __name__ == "__main__":
    main() 