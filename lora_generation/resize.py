import os
from PIL import Image

# Path to dataset
input_folder = "input/"
output_folder = "input/resized/"

os.makedirs(output_folder, exist_ok=True)

# Resize images
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        img = Image.open(os.path.join(input_folder, filename))
        img = img.resize((512, 512))  # Adjust to 1024 for SDXL
        img.save(os.path.join(output_folder, filename))

print("✅ Images resized successfully.")
