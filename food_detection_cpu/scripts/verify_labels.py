from pathlib import Path
from collections import Counter

def verify_labels():
    data_dir = Path("data")
    all_classes = set()
    
    # Process each split
    for split in ['train', 'val', 'test']:
        label_dir = data_dir / split / "labels"
        if not label_dir.exists():
            continue
        
        print(f"\nChecking {split} split...")
        files_checked = 0
        labels_found = 0
        
        # Read all label files
        for label_file in label_dir.glob("*.txt"):
            files_checked += 1
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_idx = int(parts[0])
                        all_classes.add(class_idx)
                        labels_found += 1
        
        print(f"Checked {files_checked} files")
        print(f"Found {labels_found} labels")
    
    print("\nOverall Results:")
    print(f"Found class indices: {sorted(all_classes)}")
    if max(all_classes) > 9 or min(all_classes) < 0:
        print("WARNING: Found class indices outside the expected range 0-9!")
    else:
        print("All class indices are within the expected range 0-9")

if __name__ == "__main__":
    verify_labels() 