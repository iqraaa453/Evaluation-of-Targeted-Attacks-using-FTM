import os
import csv
import pandas as pd

# Path configuration
image_dir = "data/images/images-224"
data_entry_path = "Data_Entry_2017.csv"
output_csv = "data/images/images-224.csv"

# 1. Load the NIH dataset metadata
print("Loading NIH dataset metadata...")
df = pd.read_csv(data_entry_path)
# Create a dictionary for fast lookup: { 'image_id.png': 'Finding Labels' }
label_lookup = dict(zip(df['Image Index'], df['Finding Labels']))

# 2. Define the XRV Mapping for your 3 chosen classes
# Atelectasis -> 0, Effusion -> 7, Cardiomegaly -> 10
xrv_mapping = {
    "Atelectasis": 0,
    "Effusion": 7,
    "Cardiomegaly": 10
}

# Define the Attack Target (Rotation logic)
# If True is Atelectasis (0), target is Effusion (7)
# If True is Effusion (7), target is Cardiomegaly (10)
# If True is Cardiomegaly (10), target is Atelectasis (0)
attack_rotation = {0: 7, 7: 10, 10: 0}

print("Generating medical CSV labels...")
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ImageId", "TrueLabel", "TargetClass"])

    count = 0
    for file in os.listdir(image_dir):
        if file.endswith(".png"):
            # The lookup needs the full filename like '00000001_000.png'
            findings = label_lookup.get(file, "")
            
            # Check if any of our 3 target diseases are in the findings
            true_label = None
            for disease, index in xrv_mapping.items():
                if disease in findings:
                    true_label = index
                    break # Take the first matching disease
            
            # Only add to CSV if it belongs to one of the 3 classes
            if true_label is not None:
                target_class = attack_rotation[true_label]
                image_id = file.replace(".png", "") # Save ID without extension for main.py
                writer.writerow([image_id, true_label, target_class])
                count += 1

print(f"Success! Created {output_csv} with {count} images.")
print("Mapping used: 0:Atelectasis, 7:Effusion, 10:Cardiomegaly")