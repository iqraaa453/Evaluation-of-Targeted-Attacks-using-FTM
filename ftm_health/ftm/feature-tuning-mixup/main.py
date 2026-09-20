import os
import csv
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np
import random
import copy
import gc
from PIL import Image

from config import exp_configuration
from utils import load_ground_truth, WrapperModel, load_model, EvalResult, save_images
from attacks import ftm_attack

now = datetime.now()
today_string = now.strftime("%Y-%m-%d|%H:%M:%S")

def main(args):
    exp_settings = exp_configuration[args.config_idx]

    # CPU/GPU auto-detection
    if args.device == "cuda:0" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, switching to CPU")
        device = "cpu"
    else:
        device = args.device

    batch_size = args.batch_size
    atk_type   = args.attack_method

    # ✅ Apply CLI overrides onto exp_settings
    exp_settings['ftm_beta']          = args.beta
    exp_settings['ftm_ensemble_size'] = args.ensemble_size
    if args.num_images is not None:
        exp_settings['num_images']         = args.num_images
    if args.max_iter is not None:
        exp_settings['max_iter']           = args.max_iter
    if args.target_models is not None:
        exp_settings['target_model_names'] = args.target_models
    if args.debug:
        exp_settings['num_images'] = 2 * batch_size

    print(f"Execution Time: {today_string}")
    print(f"Arguments: {args}")
    print(f"Config Settings: {exp_settings}", flush=True)
    print("#" * 50)

    # Seed
    seed = args.seed
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # ✅ Native resolution per model — used for WrapperModel resize
    # Load images at ORIGINAL resolution (e.g., 1024+), let each wrapper downscale to model's native size
    native_resolution = {
        'xrv_chexnet': 224,
        'xrv_densenet_nih': 224,
        'xrv_densenet_chex': 224,
        'xrv_densenet_pc': 224,
        'xrv_resnet50': 512,
        'ResNet50': 224,
        'DenseNet121': 224,
        'vit_base_patch16_224': 224,
        'efficientnet_b0': 224,
    }
    # Default fallback
    default_native_res = 224

    mean_rgb, std_rgb = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # ✅ FIX: XRV models expect input in [-1024, 1024] HU range.
    # Images loaded as [0, 1] floats must be mapped: output = (x - 0.5) / (1/2048)
    # This gives: 0 → -1024, 0.5 → 0, 1 → +1024  ✅ correct medical range
    mean_xrv = [0.5]
    std_xrv  = [1.0 / 2048.0]   # 0.000488 — fixes the normalization warning

    # Load at original resolution (no global resize), convert to tensor only
    trn = transforms.Compose([transforms.ToTensor()])
    image_id_list, label_ori_list, label_tar_list = load_ground_truth(args.image_csv)

    # Helper: build a WrapperModel correctly for any model name
    def make_wrapper(model_name):
        is_med    = "xrv" in model_name
        mean      = mean_xrv if is_med else mean_rgb
        std       = std_xrv  if is_med else std_rgb
        channels  = 1 if is_med else 3

        # ✅ Use model's native resolution from dict, else default
        nat_res = native_resolution.get(model_name, default_native_res)
        # WrapperModel will resize to nat_res if resize=True
        resize = True

        model_loaded = load_model(model_name)

        return WrapperModel(model_loaded, mean, std, resize, channels=channels, target_res=nat_res)

    # ── 1. Load Surrogate ────────────────────────────────────────────────────
    source_model_name = args.model_name
    is_source_medical = "xrv" in source_model_name
    print(f"Loading surrogate model: {source_model_name}")

    source_model  = make_wrapper(source_model_name).to(device)
    source_models = [source_model.eval()]

    # Get surrogate's native resolution for attack
    surrogate_native_res = native_resolution.get(source_model_name, default_native_res)
    print(f"Surrogate native resolution: {surrogate_native_res}x{surrogate_native_res}")

    if exp_settings['ftm_ensemble_size'] > 1:
        for _ in range(exp_settings['ftm_ensemble_size'] - 1):
            source_models.append(copy.deepcopy(source_model))

    # ── 2. Limit images ──────────────────────────────────────────────────────
    total_img_num  = min(len(image_id_list), exp_settings['num_images'])
    image_id_list  = list(image_id_list[:total_img_num])
    label_ori_list = list(label_ori_list[:total_img_num])
    label_tar_list = list(label_tar_list[:total_img_num])
    print(f"Images: {total_img_num}  |  Iterations: {exp_settings['max_iter']}  |  Batch: {batch_size}")

    # ── 3. Attack ────────────────────────────────────────────────────────────
    num_batches       = int(np.ceil(total_img_num / batch_size))
    all_img_batches   = []
    all_adv_imgs      = []
    all_target_labels = []

    for k in range(num_batches):
        batch_size_cur = min(batch_size, total_img_num - k * batch_size)
        num_channels   = 1 if is_source_medical else 3
        # Use surrogate's native resolution for attack
        img_batch = torch.zeros(batch_size_cur, num_channels, surrogate_native_res, surrogate_native_res).to(device)

        for i in range(batch_size_cur):
            img_name = image_id_list[k * batch_size + i]
            if not img_name.endswith('.png'):
                img_name += '.png'
            raw_img = Image.open(os.path.join(args.image_dir, img_name))
            # Load at original resolution, convert to tensor, then resize to surrogate's native res
            img_tensor = trn(raw_img.convert('L') if is_source_medical else raw_img.convert('RGB'))
            img_tensor = transforms.Resize((surrogate_native_res, surrogate_native_res), interpolation=InterpolationMode.BILINEAR)(img_tensor)
            img_batch[i] = img_tensor

        labels        = torch.tensor(label_ori_list[k*batch_size : k*batch_size+batch_size_cur]).to(device)
        target_labels = torch.tensor(label_tar_list[k*batch_size : k*batch_size+batch_size_cur]).to(device)

        gc.collect()
        adv_imgs = ftm_attack(
            source_models, img_batch, labels, target_labels,
            exp_settings, device, atk_type,
            num_iter=exp_settings['max_iter']
        )

        # Save adversarial images
        adv_save_path = os.path.join(args.save_dir, "adv_imgs")
        os.makedirs(adv_save_path, exist_ok=True)
        for i in range(batch_size_cur):
            img_id = image_id_list[k * batch_size + i]
            save_images(adv_imgs[i], os.path.join(adv_save_path, f"{img_id}_adv.png"))

        print(f"  Batch {k+1}/{num_batches} done.")

        if args.eval:
            all_img_batches.append(img_batch.cpu())
            all_adv_imgs.append(adv_imgs.cpu())
            all_target_labels.append(target_labels.cpu())

        del img_batch, adv_imgs, labels, target_labels
        gc.collect()

    del source_models, source_model
    gc.collect()
    print("\nAdversarial images generated.")

    # ── 4. Evaluation: one target model at a time ────────────────────────────
    if args.eval:
        target_model_names = exp_settings['target_model_names']
        detailed_results   = []
        print(f"Evaluating {len(target_model_names)} target models one at a time...\n")

        for model_name in target_model_names:
            print(f"  → Loading {model_name} ...", flush=True)
            is_target_medical = "xrv" in model_name

            try:
                temp_model = make_wrapper(model_name).to(device).eval()
            except Exception as e:
                print(f"  [WARNING] Skipping {model_name}: {e}")
                continue

            for k, (img_batch, adv_imgs, target_labels) in enumerate(
                    zip(all_img_batches, all_adv_imgs, all_target_labels)):

                batch_size_cur    = adv_imgs.shape[0]
                adv_imgs_dev      = adv_imgs.to(device)
                target_labels_dev = target_labels.to(device)

                with torch.no_grad():
                    test_adv = adv_imgs_dev.clone()
                    # Channel mismatch between surrogate and target
                    if is_source_medical and not is_target_medical:
                        test_adv = test_adv.repeat(1, 3, 1, 1)
                    elif not is_source_medical and is_target_medical:
                        test_adv = test_adv.mean(dim=1, keepdim=True)

                    pred_labels = torch.argmax(temp_model(test_adv), dim=1)

                    for i in range(batch_size_cur):
                        img_idx = k * batch_size + i
                        success = int(pred_labels[i].item() == target_labels_dev[i].item())
                        detailed_results.append({
                            'ImageId':   image_id_list[img_idx],
                            'ModelName': model_name,
                            'Success':   success
                        })

            del temp_model
            gc.collect()

            model_rows = [r for r in detailed_results if r['ModelName'] == model_name]
            rate = 100 * sum(r['Success'] for r in model_rows) / len(model_rows) if model_rows else 0
            print(f"     [OK] {model_name}: {rate:.1f}%  ({sum(r['Success'] for r in model_rows)}/{len(model_rows)})")

        # Save CSV
        results_dir = os.path.join(args.save_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, "attack_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=['ImageId', 'ModelName', 'Success'])
            writer.writeheader()
            writer.writerows(detailed_results)
        print(f"\nResults saved → {csv_path}")

        # ✅ Save human-readable summary to txt file
        summary_path = os.path.join(results_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("=" * 55 + "\n")
            f.write("       ATTACK RESULTS SUMMARY\n")
            f.write("=" * 55 + "\n")
            f.write(f"  Date/Time   : {today_string}\n")
            f.write(f"  Surrogate   : {args.model_name}\n")
            f.write(f"  Attack Type : {args.attack_method}\n")
            f.write(f"  Images      : {total_img_num}\n")
            f.write(f"  Iterations  : {exp_settings['max_iter']}\n")
            f.write(f"  Epsilon     : {exp_settings['epsilon']}\n")
            f.write(f"  Batch Size  : {args.batch_size}\n")
            f.write("=" * 55 + "\n")
            f.write(f"  {'Model':<28} {'Success Rate':>12}  {'Count':>8}\n")
            f.write("-" * 55 + "\n")
            overall_success = 0
            overall_total   = 0
            for name in target_model_names:
                rows = [r for r in detailed_results if r['ModelName'] == name]
                if not rows:
                    continue
                s = sum(r['Success'] for r in rows)
                t = len(rows)
                rate = 100 * s / t if t > 0 else 0
                overall_success += s
                overall_total   += t
                f.write(f"  {name:<28} {rate:>11.1f}%  ({s}/{t})\n")
            f.write("-" * 55 + "\n")
            overall_rate = 100 * overall_success / overall_total if overall_total > 0 else 0
            f.write(f"  {'OVERALL AVERAGE':<28} {overall_rate:>11.1f}%  ({overall_success}/{overall_total})\n")
            f.write("=" * 55 + "\n")

        print(f"Summary saved  → {summary_path}")

        # Also print summary to terminal
        with open(summary_path) as f:
            print(f.read())


def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",        type=str,   default="cuda:0")
    parser.add_argument("--attack_method", type=str,   default="RTMF")
    parser.add_argument("--batch_size",    type=int,   default=20)
    parser.add_argument("--model_name",    type=str,   default="xrv_chexnet")
    parser.add_argument("--image_dir",     type=str,   default="./data/images")
    parser.add_argument("--image_csv",     type=str,   default="./data/images/images-224.csv")
    parser.add_argument("--save_dir",      type=str,   default="./exp/medical_results")
    parser.add_argument("--beta",          type=float, default=0.01)
    parser.add_argument("--ensemble_size", type=int,   default=1)
    parser.add_argument("--eval",          action='store_true')
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--config_idx",    type=int,   default=1)
    parser.add_argument("--debug",         action='store_true')
    parser.add_argument("--num_images",    type=int,   default=None)
    parser.add_argument("--max_iter",      type=int,   default=None)
    parser.add_argument("--target_models", type=str,   nargs='+', default=None)
    return parser

if __name__ == "__main__":
    args = argument_parsing().parse_args()
    main(args)
    print("FINISHED")