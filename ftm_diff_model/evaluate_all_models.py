"""
Evaluate pre-generated adversarial images against all 15 target models.
Loads adversarial images from exp/test/adv_imgs/ and clean images from data/images/.
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

from utils import load_ground_truth, WrapperModel, load_model

# ── Configuration ──
ADV_DIR = "./exp/test/adv_imgs"
CLEAN_DIR = "./data/images"
CSV_FILE = "./data/images.csv"
RESULTS_DIR = "./exp/results"
IMG_SIZE = 299
DEVICE = "cpu"

ALL_TARGET_MODELS = [
    'ResNet18', 'ResNet50', 'vgg16', 'inception_v3', 'efficientnet_b0',
    'DenseNet121', 'mobilenet_v2', 'inception_resnet_v2', 'inception_v4_timm',
    'xception', 'vit_base_patch16_224', 'levit_384', 'convit_base',
    'twins_svt_base', 'pit'
]

SMALL_SIZED_MODELS = ['vit_base_patch16_224', 'levit_384', 'convit_base', 'twins_svt_base', 'pit']

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting evaluation on all 15 models...")

    # Load image metadata
    image_id_list, label_ori_list, label_tar_list = load_ground_truth(CSV_FILE)

    # Natural sort: cat.1, cat.2, ... cat.10 instead of cat.1, cat.10, cat.11
    def natural_key(s):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

    sorted_indices = sorted(range(len(image_id_list)), key=lambda i: natural_key(image_id_list[i]))
    image_id_list = [image_id_list[i] for i in sorted_indices]
    label_ori_list = [label_ori_list[i] for i in sorted_indices]
    label_tar_list = [label_tar_list[i] for i in sorted_indices]

    num_images = len(image_id_list)
    print(f"Found {num_images} images in CSV")

    # Build adversarial filename map: image_id -> adv filename
    adv_files = os.listdir(ADV_DIR)
    adv_map = {}
    for fname in adv_files:
        for img_id in image_id_list:
            if f"_{img_id}_" in fname:
                adv_map[img_id] = fname
                break
    print(f"Mapped {len(adv_map)} adversarial images")

    # Transforms
    trn = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # Pre-load all clean and adversarial images
    print("Loading images...")
    clean_imgs = []
    adv_imgs = []
    target_labels = []
    for i, img_id in enumerate(image_id_list):
        clean_path = os.path.join(CLEAN_DIR, img_id + ".png")
        adv_path = os.path.join(ADV_DIR, adv_map[img_id])

        clean_imgs.append(trn(Image.open(clean_path)).unsqueeze(0))
        adv_imgs.append(trn(Image.open(adv_path)).unsqueeze(0))
        target_labels.append(label_tar_list[i])

    clean_batch = torch.cat(clean_imgs, dim=0).to(DEVICE)
    adv_batch = torch.cat(adv_imgs, dim=0).to(DEVICE)
    target_tensor = torch.tensor(target_labels).to(DEVICE)
    print(f"Loaded {num_images} clean + adversarial image pairs")

    # Evaluate each model
    detailed_results = []
    model_summaries = []

    for model_idx, model_name in enumerate(ALL_TARGET_MODELS):
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ({model_idx+1}/15) Evaluating {model_name}...")
        try:
            is_small = model_name in SMALL_SIZED_MODELS
            model = WrapperModel(load_model(model_name), MEAN, STD, resize=is_small).to(DEVICE)
            model.eval()

            successes = 0
            with torch.no_grad():
                for i in range(num_images):
                    clean_logits = model(clean_imgs[i].to(DEVICE))
                    clean_probs = F.softmax(clean_logits, dim=1)

                    adv_logits = model(adv_imgs[i].to(DEVICE))
                    adv_probs = F.softmax(adv_logits, dim=1)
                    pred_label = torch.argmax(adv_logits, dim=1).item()

                    target_cls = target_labels[i]
                    clean_conf = clean_probs[0, target_cls].item()
                    adv_conf = adv_probs[0, target_cls].item()
                    success = 1 if pred_label == target_cls else 0
                    successes += success

                    detailed_results.append({
                        'ImageId': image_id_list[i],
                        'ModelName': model_name,
                        'CleanTargetConfidence': f"{clean_conf:.6f}",
                        'AdvTargetConfidence': f"{adv_conf:.6f}",
                        'Success': success
                    })

            rate = successes / num_images * 100
            summary = f"Model: {model_name:<25} | Attack Success: {rate:.1f}% ({successes}/{num_images})"
            model_summaries.append(summary)
            print(f"  -> {summary}")

            # Free memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  -> ERROR loading {model_name}: {e}")
            model_summaries.append(f"Model: {model_name:<25} | ERROR: {e}")
            for i in range(num_images):
                detailed_results.append({
                    'ImageId': image_id_list[i],
                    'ModelName': model_name,
                    'CleanTargetConfidence': 'N/A',
                    'AdvTargetConfidence': 'N/A',
                    'Success': 'N/A'
                })

    # ── Save Results ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
    finish_time = datetime.now().strftime("%Y-%m-%d|%H:%M:%S")

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "results_all_15_models.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            'ImageId', 'ModelName', 'CleanTargetConfidence',
            'AdvTargetConfidence', 'Success'
        ])
        writer.writeheader()
        writer.writerows(detailed_results)
    print(f"\nDetailed CSV saved to {csv_path}")

    # Summary
    valid_results = [r for r in detailed_results if r['Success'] != 'N/A']
    total = len(valid_results)
    total_successes = sum(r['Success'] for r in valid_results)
    overall_rate = total_successes / total * 100 if total > 0 else 0.0

    summary_path = os.path.join(RESULTS_DIR, "summary_all_15_models.txt")
    with open(summary_path, "w") as f:
        f.write(f"FTM Attack - Evaluation on ALL 15 Target Models ({finish_time})\n")
        f.write(f"Surrogate model: ResNet50\n")
        f.write(f"Max iterations: 300\n")
        f.write(f"Images evaluated: {num_images}\n")
        f.write(f"Target models: {ALL_TARGET_MODELS}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Success Rate: {overall_rate:.2f}% ({total_successes}/{total})\n\n")
        f.write("Per-model results:\n")
        for s in model_summaries:
            f.write(s + "\n")
    print(f"Summary saved to {summary_path}")
    print(f"\nOverall Success Rate: {overall_rate:.2f}% ({total_successes}/{total})")
    print("DONE")


if __name__ == "__main__":
    main()
