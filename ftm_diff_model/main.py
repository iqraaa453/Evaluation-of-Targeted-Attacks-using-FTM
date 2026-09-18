import os
import csv
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import random
import copy
from PIL import Image
from config import exp_configuration
from utils import load_ground_truth, WrapperModel, load_model, EvalResult, save_images
from attacks import ftm_attack

now = datetime.now()
today_string = now.strftime("%Y-%m-%d|%H:%M:%S")


def main(args):
    exp_settings = exp_configuration[args.config_idx]
    
    # ✅ CPU/GPU AUTO-DETECTION - ONLY CHANGE
    if args.device == "cuda:0" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, switching to CPU")
        device = "cpu"
    else:
        device = args.device
    
    batch_size = args.batch_size
    img_size = args.img_size
    atk_type = args.attack_method

    # update parameters
    exp_settings['ftm_beta'] = args.beta
    exp_settings['ftm_ensemble_size'] = args.ensemble_size
    if args.debug:
        exp_settings['num_images'] = 2 * args.batch_size

    print(today_string)
    print(args)
    print(exp_settings, flush=True)
    print("#" * 50)

    seed = args.seed
    random.seed(seed)
    # ✅ ONLY SET CUDA SEED IF AVAILABLE - ONLY CHANGE
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # load target models
    target_model_names = exp_settings['target_model_names']
    small_sized_models = ['vit_base_patch16_224', 'levit_384', 'convit_base', 'twins_svt_base', 'pit', 'ConvNeXt']
    mean, stddev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    trn = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), ])
    image_id_list, label_ori_list, label_tar_list = load_ground_truth(args.image_csv)
    if args.eval:
        print("Loading target models...")
        if seed != 42:
            random.seed(seed)
            c = list(zip(image_id_list, label_ori_list, label_tar_list))
            random.shuffle(c)
            image_id_list, label_ori_list, label_tar_list = zip(*c)

        target_models = [WrapperModel(load_model(x), mean, stddev, True if x in small_sized_models else False).to(device) \
                         for x in target_model_names]
        eval_results = [EvalResult(x) for x in target_model_names]
        detailed_results = []
        print("Target models loaded.", flush=True)
        print("%s target models:" % len(target_model_names), target_model_names)

    total_img_num = exp_settings['num_images']
    image_id_list = image_id_list[:total_img_num]
    label_ori_list = label_ori_list[:total_img_num]
    label_tar_list = label_tar_list[:total_img_num]

    # load source model
    source_model_name = args.model_name
    print("Loading source model", source_model_name)
    if source_model_name in small_sized_models:
        source_model = WrapperModel(load_model(source_model_name), mean, stddev, True).to(device)
    else:
        source_model = WrapperModel(load_model(source_model_name), mean, stddev).to(device)
    source_models = [source_model.eval()]
    if exp_settings['ftm_ensemble_size'] > 1:
        for _ in range(exp_settings['ftm_ensemble_size'] - 1):
            source_models.append(copy.deepcopy(source_model))
    print("Source model loaded.", flush=True)

    # perform attack
    num_batches = int(np.ceil(len(image_id_list) / batch_size))  # ✅ FIXED np.int DEPRECATION - ONLY CHANGE
    num_images = 0
    for k in range(0, num_batches):
        batch_size_cur = min(batch_size, len(image_id_list) - k * batch_size)
        img = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
        for i in range(batch_size_cur):
            img[i] = trn(Image.open(os.path.join(args.image_dir, image_id_list[k * batch_size + i] + '.png')))

        labels = torch.tensor(label_ori_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)
        target_labels = torch.tensor(label_tar_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)

        # obtain adversarial examples
        adv_imgs = ftm_attack(source_models, img, labels, target_labels, exp_settings, device, atk_type)
        num_images += batch_size_cur

        # save adversarial images
        os.makedirs(os.path.join(args.save_dir, "adv_imgs"), exist_ok=True)
        for i in range(batch_size_cur):
            img_idx = k * batch_size + i
            save_name = str(img_idx) + '_' + image_id_list[img_idx] + '_' + \
                        str(label_ori_list[img_idx]) + '_' + str(label_tar_list[img_idx]) + '.png'
            save_path = os.path.join(args.save_dir, "adv_imgs", save_name)
            save_images(adv_imgs[i], save_path)
        print(f"Saved [{num_images}/{total_img_num}] adversarial images to {args.save_dir}/adv_imgs")

        # update evaluation results
        if args.eval:
            for eval_result, target_model, model_name in zip(eval_results, target_models, target_model_names):
                target_model.eval()
                # ✅ ONLY EMPTY CACHE IF CUDA AVAILABLE - ONLY CHANGE
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    # Clean image logits and confidence
                    clean_logits = target_model(img)
                    clean_probs = F.softmax(clean_logits, dim=1)

                    # Adversarial image logits and confidence
                    adv_logits = target_model(adv_imgs)
                    adv_probs = F.softmax(adv_logits, dim=1)
                    pred_labels = torch.argmax(adv_logits, dim=1)

                    eval_result.update(pred_labels, target_labels)
                    print(eval_result)

                    # Collect per-image results
                    for i in range(batch_size_cur):
                        img_idx = k * batch_size + i
                        target_cls = target_labels[i].item()
                        clean_conf = clean_probs[i, target_cls].item()
                        adv_conf = adv_probs[i, target_cls].item()
                        success = 1 if pred_labels[i].item() == target_cls else 0
                        detailed_results.append({
                            'ImageId': image_id_list[img_idx],
                            'ModelName': model_name,
                            'CleanTargetConfidence': f"{clean_conf:.6f}",
                            'AdvTargetConfidence': f"{adv_conf:.6f}",
                            'Success': success
                        })

    # save evaluation results
    if args.eval:
        eval_results_path = os.path.join(args.save_dir, "eval_results.txt")
        finish_string = datetime.now().strftime("%Y-%m-%d|%H:%M:%S")
        with open(eval_results_path, "w") as f:
            f.write(f"Evaluation Results ({finish_string})\n")
            f.write(f"Job started at {today_string}\n")
            f.write("Arguments:\n")
            for arg, value in vars(args).items():
                f.write(f"  {arg}: {value}\n")
            f.write("-" * 50 + "\n")
            for eval_result in eval_results:
                f.write(str(eval_result) + "\n")
        print(f"Evaluation results saved to {eval_results_path}")

        # Save detailed CSV and summary with this run's outputs.
        results_dir = os.path.join(args.save_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        csv_path = os.path.join(results_dir, "results_summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                'ImageId', 'ModelName', 'CleanTargetConfidence',
                'AdvTargetConfidence', 'Success'
            ])
            writer.writeheader()
            writer.writerows(detailed_results)
        print(f"Detailed results saved to {csv_path}")

        total = len(detailed_results)
        successes = sum(r['Success'] for r in detailed_results)
        overall_rate = successes / total * 100 if total > 0 else 0.0

        modelwise_path = os.path.join(results_dir, "model_wise_summary.csv")
        with open(modelwise_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                'ModelName', 'SuccessSamples', 'TotalSamples', 'SuccessRate'
            ])
            writer.writeheader()
            for eval_result in eval_results:
                writer.writerow({
                    'ModelName': eval_result.model_name,
                    'SuccessSamples': eval_result.success_samples,
                    'TotalSamples': eval_result.total_samples,
                    'SuccessRate': f"{eval_result.success_rate:.2f}",
                })
        print(f"Model-wise summary saved to {modelwise_path}")

        summary_path = os.path.join(results_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"FTM Attack - Results Summary ({finish_string})\n")
            f.write(f"Surrogate model: {source_model_name}\n")
            f.write(f"Max iterations: {exp_settings['max_iterations']}\n")
            f.write(f"Images evaluated: {total_img_num}\n")
            f.write(f"Target models: {target_model_names}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Overall Success Rate: {overall_rate:.2f}% ({successes}/{total})\n\n")
            f.write("Per-model results:\n")
            for eval_result in eval_results:
                f.write(str(eval_result) + "\n")
        print(f"Summary saved to {summary_path}")


def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device to run the attack"
    )
    parser.add_argument(
        "--attack_method",
        type=str,
        default="RTMF",
        help="the default is RDI-TI-MI-FTM"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="batch_size as an integer"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ResNet50",
        choices=['ResNet50', 'inception_v3', 'DenseNet121', 'levit_384', 'CLIP_RN50', 'ConvNeXt'],
        help="source model name"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.01,
        help="beta parameter for FTM"
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
        help="1 for FTM, 2 for FTM-E"
    )
    parser.add_argument(
        "--eval",
        action='store_true',
        help="evaluation after attack"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./exp/test",
        help="path to save adversarial images and evaluation results"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="./data/images",
        help="path to input images"
    )
    parser.add_argument(
        "--image_csv",
        type=str,
        default="./data/images.csv",
        help="path to csv containing image info"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=299,
        help="image size"
    )
    parser.add_argument(
        "--config_idx",
        type=int,
        default=1,
        help="config containing default attack parameters"
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="debug mode"
    )
    return parser


if __name__ == "__main__":
    args = argument_parsing().parse_args()
    main(args)
    print("DONE")
