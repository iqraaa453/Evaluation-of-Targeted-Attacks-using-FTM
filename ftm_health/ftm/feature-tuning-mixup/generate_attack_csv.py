import pandas as pd
import csv

# Paths
selected_csv = "data/selection/selected_1000_images.csv"
output_csv = "data/images/images-224.csv"  # This is the attack CSV used by main.py

# XRV pathology indices
xrv_mapping = {
    "Atelectasis": 0,
    "Effusion": 7,
    "Cardiomegaly": 10
}

# Target rotation: 0→7, 7→10, 10→0
attack_rotation = {0: 7, 7: 10, 10: 0}

# Load selected 1000 images
df = pd.read_csv(selected_csv)

print(f"Loaded {len(df)} selected images")

# Generate attack CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ImageId", "TrueLabel", "TargetClass"])
    
    for _, row in df.iterrows():
        image_id = row["Image Index"].replace(".png", "")
        pathology = row["Finding Labels"]
        true_label = xrv_mapping[pathology]
        target_class = attack_rotation[true_label]
        writer.writerow([image_id, true_label, target_class])

print(f"Created attack CSV: {output_csv} with {len(df)} rows")

# Verify
df_out = pd.read_csv(output_csv)
print("\nClass distribution:")
print(df_out["TrueLabel"].value_counts().sort_index())
print("\nTarget distribution:")
print(df_out["TargetClass"].value_counts().sort_index())