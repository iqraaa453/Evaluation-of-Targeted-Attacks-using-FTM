"""
Cross-Evaluation: Test adversarial images from each surrogate against ALL models.

Tests whether adv images crafted on one surrogate transfer to:
- The other surrogates (ConvNeXt, CLIP RN50 ZS, CLIP ViT-B/32 ZS)
- The original 15 target models (already done, loaded from saved results)

Usage:
    python cross_eval.py
"""

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np


def load_ground_truth(csv_filename):
    """Load image metadata - returns lists of (image_id, true_label_0idx, target_label_0idx)"""
    image_id_list, label_ori_list, label_tar_list = [], [], []
    with open(csv_filename) as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            image_id_list.append(row['ImageId'])
            label_ori_list.append(int(row['TrueLabel']) - 1)
            label_tar_list.append(int(row['TargetClass']) - 1)
    return image_id_list, label_ori_list, label_tar_list


def load_clip_vit_b32_zeroshot():
    """Load CLIP ViT-B/32 with zero-shot ImageNet classification head."""
    import clip
    from torchvision.models import ResNet50_Weights

    clip_model, preprocess = clip.load('ViT-B/32', device='cpu')
    clip_model.eval()

    # Build zero-shot head from text embeddings
    class_names = ResNet50_Weights.IMAGENET1K_V1.meta['categories']
    prompts = [f"a photo of a {name}" for name in class_names]
    tokens = clip.tokenize(prompts)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logit_scale = clip_model.logit_scale.exp().item()

    class CLIPViTB32ZeroShot(nn.Module):
        def __init__(self, clip_model, text_features, logit_scale):
            super().__init__()
            self.clip_model = clip_model
            self.text_features = text_features  # [1000, 512]
            self.logit_scale = logit_scale

        def forward(self, x):
            # ViT-B/32 expects 224x224
            if x.shape[-1] != 224:
                x = F.interpolate(x, size=224, mode='bilinear', align_corners=False)
            img_features = self.clip_model.encode_image(x).float()
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            logits = self.logit_scale * (img_features @ self.text_features.T)
            return logits

    model = CLIPViTB32ZeroShot(clip_model, text_features, logit_scale)
    model.eval()
    return model


def load_convnext():
    """Load ConvNeXt Base with ImageNet weights."""
    import torchvision
    model = torchvision.models.convnext_base(weights='IMAGENET1K_V1')
    model.eval()
    return model


def load_clip_rn50_zeroshot():
    """Load CLIP RN50 with zero-shot head (same as CLIPResNet50ForFTM)."""
    from utils import CLIPResNet50ForFTM
    model = CLIPResNet50ForFTM()
    model.eval()
    return model


def load_resnet50():
    """Load standard ResNet50 with ImageNet weights."""
    import torchvision
    model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
    model.eval()
    return model


def evaluate_adv_images(adv_dir, model, target_labels_dict, model_name, img_size=299):
    """Evaluate adversarial images against a model.
    
    Args:
        adv_dir: path to adversarial images
        model: PyTorch model
        target_labels_dict: dict mapping image_id -> target_label (0-indexed)
        model_name: name for display
        img_size: input size for model (224 for CLIP/ViT, 299 for others)
    
    Returns:
        (success_count, total_count)
    """
    transform = transforms.ToTensor()
    success = 0
    total = 0

    adv_files = sorted([f for f in os.listdir(adv_dir) if f.endswith('.png')])

    for fname in adv_files:
        # Filename format: {index}_{imageId}_{trueLabel}_{targetLabel}.png
        # Examples: "0_cat.1_281_207.png" or "0_0c7ac4a8c9dfa802_305_778.png"
        # Last two _-separated parts are always labels; first part is index
        # Image ID is everything between first _ and second-to-last _
        parts = fname.replace('.png', '').split('_')
        # parts[-1] = targetLabel, parts[-2] = trueLabel, parts[0] = index
        # image_id = parts[1:-2] joined by _ (handles IDs with underscores)
        image_id = '_'.join(parts[1:-2])

        if image_id not in target_labels_dict:
            continue

        target_label = target_labels_dict[image_id]
        img = Image.open(os.path.join(adv_dir, fname)).convert('RGB')
        x = transform(img).unsqueeze(0)

        # Resize if needed
        if x.shape[-1] != img_size:
            x = F.interpolate(x, size=img_size, mode='bilinear', align_corners=False)

        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(1).item()

        if pred == target_label:
            success += 1
        total += 1

    return success, total


def main():
    csv_file = './data/images.csv'
    image_ids, true_labels, target_labels = load_ground_truth(csv_file)

    # Build dict: image_id -> target_label (0-indexed)
    target_dict = {}
    for img_id, tgt in zip(image_ids, target_labels):
        target_dict[img_id] = tgt

    # Adversarial image directories from each surrogate
    adv_dirs = {
        'ResNet50': './exp/ResNet50/adv_imgs',
        'ConvNeXt': './exp/convnext/adv_imgs',
        'CLIP_RN50_ZS': './exp/clip_rn50_zeroshot/adv_imgs',
    }

    # Target models to evaluate against (the 3 surrogates + CLIP ViT-B/32)
    print("Loading evaluation models...")
    print("  Loading ResNet50...")
    resnet50_model = load_resnet50()
    print("  Loading ConvNeXt Base...")
    convnext_model = load_convnext()
    print("  Loading CLIP RN50 (zero-shot)...")
    clip_rn50_model = load_clip_rn50_zeroshot()
    print("  Loading CLIP ViT-B/32 (zero-shot)...")
    clip_vit_model = load_clip_vit_b32_zeroshot()
    print("All models loaded.\n")

    eval_models = {
        'ResNet50': (resnet50_model, 224),
        'ConvNeXt': (convnext_model, 224),
        'CLIP_RN50_ZS': (clip_rn50_model, 224),  # internally resizes to 224
        'CLIP_ViT-B/32_ZS': (clip_vit_model, 224),
    }

    # Results matrix: adv_source -> eval_target -> (success, total)
    results = {}

    for src_name, adv_dir in adv_dirs.items():
        if not os.path.exists(adv_dir):
            print(f"WARNING: {adv_dir} not found, skipping {src_name}")
            continue

        num_adv = len([f for f in os.listdir(adv_dir) if f.endswith('.png')])
        print(f"=== Evaluating {src_name} adversarial images ({num_adv} images) ===")
        results[src_name] = {}

        for tgt_name, (model, img_size) in eval_models.items():
            success, total = evaluate_adv_images(
                adv_dir, model, target_dict, tgt_name, img_size
            )
            rate = success / total * 100 if total > 0 else 0
            results[src_name][tgt_name] = (success, total, rate)
            print(f"  {src_name} -> {tgt_name}: {rate:.1f}% ({success}/{total})")

        print()

    # Print summary table
    print("=" * 75)
    print("CROSS-EVALUATION SUMMARY")
    print("=" * 75)
    eval_order = ['ResNet50', 'ConvNeXt', 'CLIP_RN50_ZS', 'CLIP_ViT-B/32_ZS']
    header = f"{'Adv Source':<20}" + " | ".join(f"{t:<18}" for t in eval_order)
    print(header)
    print("-" * 100)
    for src_name in adv_dirs:
        if src_name not in results:
            continue
        r = results[src_name]
        cells = []
        for tgt in eval_order:
            if tgt in r:
                s, t, rate = r[tgt]
                cells.append(f"{rate:.1f}% ({s}/{t})")
            else:
                cells.append("N/A")
        print(f"{src_name:<20}" + " | ".join(f"{c:<18}" for c in cells))
    print("=" * 100)

    # Save results
    output_file = './exp/results/cross_eval_results.txt'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("Cross-Evaluation: Adversarial Transferability Between Surrogates\n")
        f.write("=" * 75 + "\n")
        f.write("Date: 2026-03-03\n")
        f.write("Task: Do adv images from one surrogate fool other models?\n")
        f.write("Attack: RDI-TI-MI-FTM (RTMF) | eps=16/255, alpha=2/255, iter=300\n")
        f.write("=" * 75 + "\n\n")
        f.write("Rows = surrogate that generated the adversarial images\n")
        f.write("Cols = model being evaluated (target)\n\n")
        eval_order = ['ResNet50', 'ConvNeXt', 'CLIP_RN50_ZS', 'CLIP_ViT-B/32_ZS']
        f.write(f"{'Adv Source':<20} | " + " | ".join(f"{t:<18}" for t in eval_order) + "\n")
        f.write("-" * 100 + "\n")
        for src_name in adv_dirs:
            if src_name not in results:
                continue
            r = results[src_name]
            cells = []
            for tgt in eval_order:
                if tgt in r:
                    s, t, rate = r[tgt]
                    cells.append(f"{rate:.1f}% ({s}/{t})")
                else:
                    cells.append("N/A")
            f.write(f"{src_name:<20} | " + " | ".join(f"{c:<18}" for c in cells) + "\n")
        f.write("=" * 100 + "\n")

    print(f"\nResults saved to {output_file}")
    print("DONE")


if __name__ == '__main__':
    main()
