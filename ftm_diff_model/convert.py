from PIL import Image
import os

image_dir = "data/images"

for file in os.listdir(image_dir):
    if file.endswith(".jpg"):
        img = Image.open(os.path.join(image_dir, file))
        new_name = file.replace(".jpg", ".png")
        img.save(os.path.join(image_dir, new_name))
        os.remove(os.path.join(image_dir, file))

print("Conversion complete")