import os
import csv

image_dir = "data/images"

with open("data/images.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ImageId", "TrueLabel", "TargetClass"])

    for file in os.listdir(image_dir):
        if file.endswith(".png"):
            image_id = file.replace(".png", "")

            if "cat" in file.lower():
                true_label = 282
                target = 208
            else:
                true_label = 208
                target = 282

            writer.writerow([image_id, true_label, target])

print("CSV created")