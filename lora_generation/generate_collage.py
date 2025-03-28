from PIL import Image

# Folder where your images are saved
image_folder = "./output"  # or wherever your images are
image_paths = [f"{image_folder}/image_{i}.png" for i in range(10)]

# Load images
images = [Image.open(p) for p in image_paths]

# Resize all to the same size (optional, just in case)
thumb_size = (images[0].width, images[0].height)
images = [img.resize(thumb_size) for img in images]

# Grid size: 5 columns x 2 rows
cols = 5
rows = 2
grid_width = cols * thumb_size[0]
grid_height = rows * thumb_size[1]

# Create the grid canvas
grid_img = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))

# Paste each image into the grid
for index, img in enumerate(images):
    x = (index % cols) * thumb_size[0]
    y = (index // cols) * thumb_size[1]
    grid_img.paste(img, (x, y))

# Save the result
grid_img.save("output/image_grid.png")
print("Saved grid to image_grid.png")
