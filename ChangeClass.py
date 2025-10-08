import os
from glob import glob

# Path to your labels folder (change this!)
label_dir = "test/labels"

# Get all .txt files recursively
txt_files = glob(os.path.join(label_dir, "**", "*.txt"), recursive=True)

for txt_path in txt_files:
    with open(txt_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        # Replace the first number (class id) with 0
        parts[0] = "0"
        new_lines.append(" ".join(parts) + "\n")

    with open(txt_path, "w") as f:
        f.writelines(new_lines)
