"""
Evaluate pre-generated adversarial images against medical (XRV) target models.
Loads adversarial .pt tensors (if available) or PNG images from exp/medical_results/adv_imgs/
and clean images from data/images/.
Saves detailed CSV and summary to exp/results/.
"""

import os
import csv
import re
from datetime import datetime

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Ensure these are imported from your updated utils.py
from utils import load_ground_truth, WrapperModel, load_model

# ── Configuration ──
ADV_DIR = "./exp/medical_results/adv_imgs"
CLEAN_DIR = "./data/images/images-224"
CSV_FILE = "./data/images/images-224.csv"
RESULTS_DIR = "./exp/results"
IMG_SIZE = 224  # Must match the img_size used during attack generation in main.py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Updated to Medical-Specific Target Models
ALL_TARGET_MODELS = [
    'xrv_chexnet',        # Surrogate model - verify attack works on it
    'DenseNet121',        # Standard ImageNet version (for cross-domain transfer test)
    'ResNet50',           # Standard ImageNet version
    'vit_base_patch16_224'# Transformer-based ImageNet model
]

# Models that strictly require 224x224 or XRV processing
SMALL_SIZED_MODELS = ['vit_base_patch16_224', 'xrv_chexnet', 'xrv_densenet_nih', 'xrv_densenet_chex', 'xrv_densenet_pc']

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# For XRV models (grayscale) - matching main.py
MEAN_XRV = [0.5]
STD_XRV = [0.5]


def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Medical Evaluation on {len(ALL_TARGET_MODELS)} models...")
    print(f"Using Device: {DEVICE}")

    # Load image metadata
    image_id_list, label_ori_list, label_tar_list = load_ground_truth(CSV_FILE)

    # Natural sort logic
    def natural_key(s):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

    sorted_indices = sorted(range(len(image_id_list)), key=lambda i: natural_key(image_id_list[i]))
    image_id_list = [image_id_list[i] for i in sorted_indices]
    label_ori_list = [label_ori_list[i] for i in sorted_indices]
    label_tar_list = [label_tar_list[i] for i in sorted_indices]

    num_images = len(image_id_list)
    print(f"Found {num_images} images in CSV")

    # Build adversarial filename map (prefer .pt tensors over .png)
    adv_files = os.listdir(ADV_DIR)
    adv_map = {}      # img_id -> png filename
    adv_pt_map = {}   # img_id -> .pt filename (lossless)
    for fname in adv_files:
        for img_id in image_id_list:
            if img_id in fname:
                if fname.endswith('.pt'):
                    adv_pt_map[img_id] = fname
                elif fname.endswith('.png'):
                    adv_map[img_id] = fname
                break
    # Merge: an image is available if it has either .pt or .png
    all_adv_ids = set(adv_map.keys()) | set(adv_pt_map.keys())
    print(f"Mapped {len(all_adv_ids)} adversarial images ({len(adv_pt_map)} .pt, {len(adv_map)} .png)")

    # Transforms
    trn = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # Pre-load all clean and adversarial images
    print("Loading images into memory...")
    clean_imgs = []
    adv_imgs = []
    target_labels = []
    evaluated_image_ids = []
    for i, img_id in enumerate(image_id_list):
        if img_id not in all_adv_ids:
            continue
        clean_path = os.path.join(CLEAN_DIR, img_id + ".png")

        # Load clean image as RGB (WrapperModel handles channel conversion)
        clean_imgs.append(trn(Image.open(clean_path).convert('RGB')).unsqueeze(0))

        # Prefer lossless .pt tensor; fall back to PNG
        if img_id in adv_pt_map:
            adv_tensor = torch.load(os.path.join(ADV_DIR, adv_pt_map[img_id]), map_location='cpu')
            if adv_tensor.dim() == 3:  # [C, H, W] -> [1, C, H, W]
                adv_tensor = adv_tensor.unsqueeze(0)
            adv_imgs.append(adv_tensor)
        else:
            adv_path = os.path.join(ADV_DIR, adv_map[img_id])
            adv_imgs.append(trn(Image.open(adv_path).convert('RGB')).unsqueeze(0))

        target_labels.append(label_tar_list[i])
        evaluated_image_ids.append(img_id)

    num_images = len(clean_imgs)
    print(f"Evaluating {num_images} images with adversarial counterparts")

    # Evaluate each model
    detailed_results = []
    model_summaries = []

    for model_idx, model_name in enumerate(ALL_TARGET_MODELS):
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ({model_idx+1}/{len(ALL_TARGET_MODELS)}) Evaluating {model_name}...")

        # attempt to load the base network, catch weight‑size mismatches
        try:
            base_model = load_model(model_name)
        except Exception as e:
            print(f"  ! SKIPPING {model_name} (load failed): {e}")
            continue

        # configure wrapper parameters based on whether this is an xrv (grayscale) model
        is_xrv = "xrv" in model_name
        if is_xrv:
            mean = MEAN_XRV
            std = STD_XRV
            channels = 1
        else:
            mean = MEAN
            std = STD
            channels = 3
        is_small = model_name in SMALL_SIZED_MODELS or is_xrv

        try:
            model = WrapperModel(base_model, mean, std, resize=is_small, channels=channels).to(DEVICE)
            model.eval()
        except Exception as e:
            print(f"  ! SKIPPING {model_name} (wrapper init failed): {e}")
            continue

        successes = 0
        invalid_targets = 0
        try:
            with torch.no_grad():
                for i in range(num_images):
                    clean_input = clean_imgs[i].to(DEVICE)
                    adv_input = adv_imgs[i].to(DEVICE)

                    clean_logits = model(clean_input)
                    clean_probs = F.softmax(clean_logits, dim=1)

                    adv_logits = model(adv_input)
                    adv_probs = F.softmax(adv_logits, dim=1)
                    pred_label = torch.argmax(adv_logits, dim=1).item()

                    target_cls = target_labels[i]

                    # Handle cases where the target label is not within the model's output space
                    if target_cls >= adv_probs.shape[1]:
                        # record for summary but skip when computing success rate
                        invalid_targets += 1
                        clean_conf, adv_conf, success = 0.0, 0.0, 0
                    else:
                        clean_conf = clean_probs[0, target_cls].item()
                        adv_conf = adv_probs[0, target_cls].item()
                        # Success: only when predicted label matches target class
                        success = 1 if pred_label == target_cls else 0
                        successes += success

                    detailed_results.append({
                        'ImageId': evaluated_image_ids[i],
                        'ModelName': model_name,
                        'CleanTargetConfidence': f"{clean_conf:.6f}",
                        'AdvTargetConfidence': f"{adv_conf:.6f}",
                        'Success': success
                    })

            valid_images = num_images - invalid_targets
            if valid_images > 0:
                rate = successes / valid_images * 100
            else:
                rate = 0.0
            summary = f"Model: {model_name:<25} | Attack Success: {rate:.1f}% ({successes}/{valid_images})"
            if invalid_targets > 0:
                summary += f"  [skipped {invalid_targets} out-of-range targets]"
            model_summaries.append(summary)
            print(f"  -> {summary}")

            # Free memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  -> ERROR evaluating {model_name}: {e}")
            model_summaries.append(f"Model: {model_name:<25} | ERROR: {e}")

    # ── Save Results ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
    finish_time = datetime.now().strftime("%Y-%m-%d|%H:%M:%S")

    csv_path = os.path.join(RESULTS_DIR, "medical_evaluation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            'ImageId', 'ModelName', 'CleanTargetConfidence',
            'AdvTargetConfidence', 'Success'
        ])
        writer.writeheader()
        writer.writerows(detailed_results)

    valid_results = [r for r in detailed_results if r['Success'] != 'N/A']
    total = len(valid_results)
    total_successes = sum(r['Success'] for r in valid_results if isinstance(r['Success'], int))
    overall_rate = total_successes / total * 100 if total > 0 else 0.0

    summary_path = os.path.join(RESULTS_DIR, "medical_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"FTM Medical Attack Evaluation ({finish_time})\n")
        f.write(f"Surrogate model: xrv_chexnet\n")
        f.write(f"Images evaluated: {num_images}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Success Rate: {overall_rate:.2f}% ({total_successes}/{total})\n\n")
        for s in model_summaries:
            f.write(s + "\n")
            
    print(f"\nEvaluation Complete. Overall Success Rate: {overall_rate:.2f}%")
    print(f"Detailed results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()