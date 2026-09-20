import os
import csv
import pandas as pd
import random

# Path configuration
image_dir = "data/images/images-224"
data_entry_path = "Data_Entry_2017.csv"
output_csv = "data/images/images-224.csv"

# ============================================================
# CONFIGURATION: Change this to select exactly N images
# ============================================================
TARGET_NUM_IMAGES = 1000  # <-- Set to 1000 (or any number <= available)
RANDOM_SEED = 42          # <-- Fixed seed for reproducible selection
# ============================================================

# 1. Load the NIH dataset metadata
print("Loading NIH dataset metadata...")
df = pd.read_csv(data_entry_path)
label_lookup = dict(zip(df['Image Index'], df['Finding Labels']))

# 2. Define the XRV Mapping for 3 selected classes
# XRV pathology indices: Atelectasis=0, Effusion=7, Cardiomegaly=10
xrv_mapping = {
    "Atelectasis": 0,
    "Effusion": 7,
    "Cardiomegaly": 10
}

# 3. Define the Attack Target (Rotation logic for targeted attack)
# True label 0 (Atelectasis) -> Target 7 (Effusion)
# True label 7 (Effusion)    -> Target 10 (Cardiomegaly)
# True label 10 (Cardiomegaly) -> Target 0 (Atelectasis)
attack_rotation = {0: 7, 7: 10, 10: 0}

print("Generating medical CSV labels...")

# --- Step A: Collect ALL eligible images (single-label from 3 classes) ---
eligible_by_class = {0: [], 7: [], 10: []}

for file in os.listdir(image_dir):
    if not file.endswith(".png"):
        continue
    findings = label_lookup.get(file, "")
    
    matched = []
    for disease, index in xrv_mapping.items():
        if disease in findings:
            matched.append((disease, index))
    
    # SELECTION CRITERIA: EXACTLY ONE of the 3 target diseases
    if len(matched) == 1:
        true_label = matched[0][1]
        target_class = attack_rotation[true_label]
        image_id = file.replace(".png", "")
        eligible_by_class[true_label].append((image_id, true_label, target_class))

# Count per class
for lbl in [0, 7, 10]:
    print(f"  Class {lbl}: {len(eligible_by_class[lbl])} eligible images")

total_eligible = sum(len(v) for v in eligible_by_class.values())
print(f"Total eligible: {total_eligible}")

# --- Step B: Stratified sampling to get exactly TARGET_NUM_IMAGES ---
random.seed(RANDOM_SEED)

# Calculate how many per class (proportional to availability)
selected = []
if TARGET_NUM_IMAGES >= total_eligible:
    # Use all available
    for lst in eligible_by_class.values():
        selected.extend(lst)
    print(f"Requested {TARGET_NUM_IMAGES} but only {total_eligible} available. Using all.")
else:
    # Proportional allocation
    for lbl in [0, 7, 10]:
        n_class = len(eligible_by_class[lbl])
        # Proportional share
        share = int(round(TARGET_NUM_IMAGES * n_class / total_eligible))
        share = min(share, n_class)  # cap at available
        picked = random.sample(eligible_by_class[lbl], share)
        selected.extend(picked)
        print(f"  Picked {len(picked)} from class {lbl} (available: {n_class})")

# If rounding caused mismatch, adjust from largest class
while len(selected) > TARGET_NUM_IMAGES:
    selected.pop()
while len(selected) < TARGET_NUM_IMAGES:
    # Add from class with most remaining
    remaining = {lbl: [x for x in eligible_by_class[lbl] if x not in selected] for lbl in [0,7,10]}
    largest_lbl = max(remaining, key=lambda k: len(remaining[k]))
    if remaining[largest_lbl]:
        selected.append(remaining[largest_lbl].pop())
    else:
        break

random.shuffle(selected)
print(f"Final selection: {len(selected)} images")

# --- Step C: Write CSV ---
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ImageId", "TrueLabel", "TargetClass"])
    for row in selected:
        writer.writerow(row)

print(f"Success! Created {output_csv} with {len(selected)} images.")
print("Selection: Stratified random sampling (seed=42) across 3 classes")
print("Criteria: Single-label only (Atelectasis=0, Effusion=7, Cardiomegaly=10)")
print("Target rotation: 0→7→10→0")