import os
from PIL import Image

# --- Configuration ---
input_folder = "train/images"   # Change this to your folder path
output_folder = input_folder           # Or use a different folder if you prefer

# --- Conversion ---
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".png"):
        png_path = os.path.join(input_folder, filename)
        jpg_filename = os.path.splitext(filename)[0] + ".jpg"
        jpg_path = os.path.join(output_folder, jpg_filename)

        # Open the PNG image and convert to RGB (required for JPEG)
        with Image.open(png_path) as img:
            os.remove(png_path)

        # print(f"Converted: {filename} -> {jpg_filename}")