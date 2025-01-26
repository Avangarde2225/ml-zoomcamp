import os
from pathlib import Path
import shutil

# Define the mapping from old class indices to new ones
class_mapping = {
    # Pizza and related items (0)
    41: 0, 10: 0, 11: 0, 51: 0,
    
    # Burger and related items (1)
    42: 1, 13: 1, 52: 1,
    
    # Sandwich and related items (2)
    43: 2, 14: 2, 53: 2,
    
    # Salad and related items (3)
    44: 3, 15: 3, 54: 3,
    
    # Sushi and related items (4)
    45: 4, 16: 4, 55: 4,
    
    # Pasta and related items (5)
    46: 5, 17: 5, 56: 5,
    
    # Steak and related items (6)
    47: 6, 18: 6, 57: 6,
    
    # Soup and related items (7)
    48: 7, 19: 7, 58: 7,
    
    # Taco and related items (8)
    49: 8, 21: 8, 59: 8,
    
    # Curry and related items (9)
    50: 9, 22: 9, 60: 9,
    
    # Additional mappings for remaining classes
    23: 0, 25: 1, 26: 2, 27: 3, 28: 4, 29: 5,
    32: 6, 33: 7, 34: 8, 35: 9, 36: 0, 39: 1,
    40: 2, 61: 3, 62: 4, 63: 5, 64: 6, 65: 7,
    66: 8, 67: 9, 68: 0, 69: 1, 70: 2, 71: 3,
    72: 4, 73: 5, 74: 6, 75: 7, 76: 8, 77: 9,
    79: 0
}

def fix_label_file(file_path):
    """Fix class indices in a single label file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        old_class = int(parts[0])
        if old_class in class_mapping:
            new_class = class_mapping[old_class]
            fixed_lines.append(f"{new_class} {' '.join(parts[1:])}\n")
    
    if fixed_lines:
        with open(file_path, 'w') as f:
            f.writelines(fixed_lines)

def main():
    # Process each split
    data_dir = Path("data")
    for split in ['train', 'val', 'test']:
        label_dir = data_dir / split / "labels"
        if not label_dir.exists():
            continue
        
        print(f"Processing {split} split...")
        for label_file in label_dir.glob("*.txt"):
            fix_label_file(label_file)
        print(f"Finished processing {split} split")

if __name__ == "__main__":
    main() 