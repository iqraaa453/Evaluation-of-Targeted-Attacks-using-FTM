import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# Project root
PROJECT_ROOT = Path(
    r"D:\ftm\ftm_health\ftm\feature-tuning-mixup"
)

# NIH metadata CSV
METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "Data_Entry_2017_v2020.csv"
)

# Output directory
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "selection"
)

# Output files
ELIGIBLE_FILE = (
    OUTPUT_DIR
    / "eligible_images.csv"
)

SELECTED_FILE = (
    OUTPUT_DIR
    / "selected_1000_images.csv"
)

# Random seed for reproducibility
SEED = 42

# Exactly these three NIH pathologies
CLASSES = [
    "Atelectasis",
    "Effusion",
    "Cardiomegaly"
]

# ============================================================
# STRATIFIED SAMPLE COUNTS
# ============================================================
#
# Total = 334 + 333 + 333 = 1000
#
# Atelectasis   -> 334
# Effusion      -> 333
# Cardiomegaly  -> 333
#
# ============================================================

SAMPLE_COUNTS = {
    "Atelectasis": 334,
    "Effusion": 333,
    "Cardiomegaly": 333,
}

# ============================================================
# CHECK PATHS
# ============================================================

print("=" * 70)
print("NIH DATASET STRATIFIED SAMPLING")
print("=" * 70)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("Metadata file:")
print(METADATA_PATH)

# Check whether metadata exists
if not METADATA_PATH.exists():

    print()
    print("ERROR: Metadata CSV was not found!")
    print()
    print("Expected location:")
    print(METADATA_PATH)
    print()

    raise FileNotFoundError(
        f"Could not find NIH metadata file:\n{METADATA_PATH}"
    )

print()
print("Metadata file found successfully.")

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print()
print("Output directory:")
print(OUTPUT_DIR)

# ============================================================
# LOAD NIH METADATA
# ============================================================

print()
print("=" * 70)
print("LOADING NIH METADATA")
print("=" * 70)

df = pd.read_csv(
    METADATA_PATH
)

print()
print("Total NIH records:", len(df))

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Image Index",
    "Finding Labels"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise RuntimeError(
        "Required columns are missing from the NIH metadata: "
        + ", ".join(missing_columns)
    )

print()
print("Required columns found:")
for column in required_columns:
    print("  -", column)

# ============================================================
# STEP 1:
# KEEP ONLY EXACTLY SINGLE-LABEL IMAGES
# ============================================================

print()
print("=" * 70)
print("STEP 1: FILTERING SINGLE-LABEL IMAGES")
print("=" * 70)

# A multi-label image has labels separated by "|"
#
# Example:
# Pneumonia|Effusion
#
# We remove these and keep only images with exactly one label.

single_label = df[
    ~df["Finding Labels"].str.contains(
        r"\|",
        regex=True,
        na=False
    )
].copy()

print()
print("Single-label records:", len(single_label))

# ============================================================
# STEP 2:
# KEEP ONLY THE THREE REQUIRED PATHOLOGIES
# ============================================================

print()
print("=" * 70)
print("STEP 2: FILTERING REQUIRED PATHOLOGIES")
print("=" * 70)

eligible = single_label[
    single_label["Finding Labels"].isin(CLASSES)
].copy()

# Reset index
eligible = eligible.reset_index(drop=True)

print()
print("Eligible images by class:")

class_counts = (
    eligible["Finding Labels"]
    .value_counts()
    .reindex(CLASSES, fill_value=0)
)

for pathology in CLASSES:

    print(
        f"  {pathology:<15} : "
        f"{class_counts[pathology]}"
    )

print()
print("Total eligible:", len(eligible))

# ============================================================
# VERIFY THAT ENOUGH IMAGES EXIST
# ============================================================

print()
print("=" * 70)
print("CHECKING SAMPLE AVAILABILITY")
print("=" * 70)

for pathology, sample_count in SAMPLE_COUNTS.items():

    available = int(
        class_counts[pathology]
    )

    print()
    print(
        f"{pathology}: "
        f"{available} available, "
        f"{sample_count} required"
    )

    if available < sample_count:

        raise RuntimeError(
            f"Not enough {pathology} images. "
            f"Available: {available}, "
            f"Required: {sample_count}"
        )

print()
print("All required sample sizes are available.")

# ============================================================
# SAVE COMPLETE ELIGIBLE POPULATION
# ============================================================

eligible.to_csv(
    ELIGIBLE_FILE,
    index=False
)

print()
print("Complete eligible population saved to:")
print(ELIGIBLE_FILE)

# ============================================================
# STEP 3:
# STRATIFIED RANDOM SAMPLING
# ============================================================

print()
print("=" * 70)
print("STEP 3: STRATIFIED RANDOM SAMPLING")
print("=" * 70)

print()
print("Random seed:", SEED)

selected_parts = []

for pathology, sample_count in SAMPLE_COUNTS.items():

    print()
    print("-" * 50)
    print("Sampling:", pathology)

    subset = eligible[
        eligible["Finding Labels"] == pathology
    ].copy()

    available = len(subset)

    print("Available:", available)
    print("Selecting:", sample_count)

    # Random sampling within this pathology
    sampled = subset.sample(
        n=sample_count,
        random_state=SEED
    )

    print("Selected:", len(sampled))

    selected_parts.append(
        sampled
    )

# ============================================================
# COMBINE THE THREE STRATA
# ============================================================

selected = pd.concat(
    selected_parts,
    ignore_index=True
)

# ============================================================
# VERIFY TOTAL
# ============================================================

if len(selected) != 1000:

    raise RuntimeError(
        f"Expected exactly 1000 images, "
        f"but selected {len(selected)}."
    )

# ============================================================
# SHUFFLE FINAL DATASET
# ============================================================

selected = selected.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)

# ============================================================
# ADD SAMPLE ID
# ============================================================

selected.insert(
    0,
    "Sample_ID",
    range(
        1,
        len(selected) + 1
    )
)

# ============================================================
# SAVE FINAL 1000 IMAGES
# ============================================================

selected.to_csv(
    SELECTED_FILE,
    index=False
)

# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("FINAL SAMPLE")
print("=" * 70)

print()
print("Total selected:", len(selected))

print()
print("Selected images by pathology:")

final_counts = (
    selected["Finding Labels"]
    .value_counts()
    .reindex(CLASSES, fill_value=0)
)

for pathology in CLASSES:

    print(
        f"  {pathology:<15} : "
        f"{final_counts[pathology]}"
    )

# ============================================================
# VERIFY EXACT STRATIFICATION
# ============================================================

expected_total = sum(
    SAMPLE_COUNTS.values()
)

actual_total = len(selected)

if actual_total != expected_total:

    raise RuntimeError(
        f"Stratification verification failed. "
        f"Expected {expected_total}, "
        f"got {actual_total}."
    )

for pathology in CLASSES:

    expected = SAMPLE_COUNTS[pathology]
    actual = final_counts[pathology]

    if actual != expected:

        raise RuntimeError(
            f"Incorrect sample count for {pathology}. "
            f"Expected {expected}, got {actual}."
        )

print()
print("Stratification verified successfully.")

# ============================================================
# OUTPUT LOCATION
# ============================================================

print()
print("Final 1000-image manifest saved to:")

print(
    SELECTED_FILE
)

# ============================================================
# PREVIEW
# ============================================================

print()
print("=" * 70)
print("FIRST 10 SELECTED IMAGES")
print("=" * 70)

preview_columns = [
    "Sample_ID",
    "Image Index",
    "Finding Labels"
]

print(
    selected[
        preview_columns
    ].head(10).to_string(
        index=False
    )
)

# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("SAMPLING COMPLETE")
print("=" * 70)

print()
print("1000 NIH images selected using:")
print("  - Single-label filtering")
print("  - 3 selected pathologies")
print("  - Stratified random sampling")
print("  - Random seed = 42")
print("  - Total = 1000 images")

print()
print("Next file to use:")
print(SELECTED_FILE)

print()
print("=" * 70)