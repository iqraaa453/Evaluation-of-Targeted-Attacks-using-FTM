import os
import csv
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
import torchvision
import timm
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F

# NEW: Import for medical models
try:
    import torchxrayvision as xrv
except ImportError:
    print("Please install torchxrayvision: pip install torchxrayvision")

# Simple wrapper model to normalize an input image
class WrapperModel(nn.Module):
    def __init__(self, model, mean, std, resize=False, channels=3, target_res=224):
        super(WrapperModel, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
        self.model = model
        self.resize = resize
        self.channels = channels
        self.target_res = target_res

    def forward(self, x):
        if self.channels == 1 and x.shape[1] == 3:
            # Convert RGB to grayscale
            x = 0.2989 * x[:, 0] + 0.5870 * x[:, 1] + 0.1140 * x[:, 2]
            x = x.unsqueeze(1)
        elif self.channels == 3 and x.shape[1] == 1:
            # Convert grayscale to RGB by repeating the single channel
            x = x.repeat(1, 3, 1, 1)
        
        if self.resize:
            # Resize to model's native resolution
            x = transforms.Resize((self.target_res, self.target_res), interpolation=InterpolationMode.BILINEAR)(x)
        
        # Normalization logic: (input - mean) / std
        x_norm = (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]
        return self.model(x_norm)

    def normalize(self, x):
        if self.resize:
            x = transforms.Resize((self.target_res, self.target_res), interpolation=InterpolationMode.BILINEAR)(x)
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


def load_model(model_name):
    """
    Loads both Medical (XRV) and Standard (timm/torchvision) models.
    """
    # 1. MEDICAL MODELS (TorchXRayVision)
    medical_mapping = {
        "xrv_chexnet": "densenet121-res224-all",      # Best Surrogate
        "xrv_densenet_nih": "densenet121-res224-nih",
        "xrv_densenet_chex": "densenet121-res224-chex",
        "xrv_densenet_pc": "densenet121-res224-pc",
        "xrv_resnet50": "resnet50-res512-all",
    }

    if model_name in medical_mapping:
        # Load the XRV model from library
        model = xrv.models.get_model(medical_mapping[model_name])
        
        # IMPORTANT: Disable internal thresholds to get raw logits
        # This is required for gradient-based attacks like FTM
        model.op_threshs = None 
        return model

    # 2. STANDARD MODELS (For comparison)
    elif model_name == "ResNet50":
        model = torchvision.models.resnet50(pretrained=True)
    elif model_name == 'DenseNet121':
        model = torchvision.models.densenet121(pretrained=True)
    elif model_name == "vit_base_patch16_224":
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
    elif model_name == "efficientnet_b0":
        model = timm.create_model("efficientnet_b0", pretrained=True)
    else:
        # If you see this error, add the model name to this list or config.py
        raise ValueError(f"Not supported model name: {model_name}")
    
    return model

def load_ground_truth(csv_filename):
    """
    Loads metadata from the CSV. 
    Note: Removed '- 1' because medical labels (0-17) are already 0-indexed.
    """
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append(row['ImageId'])
            # We use the raw integer from the CSV
            label_ori_list.append(int(row['TrueLabel']))
            label_tar_list.append(int(row['TargetClass']))

    return image_id_list, label_ori_list, label_tar_list


class EvalResult:
    def __init__(self, model_name):
        self.model_name = model_name
        self.total_samples = 0
        self.success_samples = 0
        self.success_rate = 0.0

    def update(self, pred_labels, target_labels):
        batch_size = len(pred_labels)
        self.total_samples += batch_size
        # Success = how many times the model predicted the TARGET class
        self.success_samples += torch.sum(pred_labels == target_labels).item()
        self.success_rate = self.success_samples / self.total_samples * 100

    def get_results(self):
        return {
            "model_name": self.model_name,
            "total_samples": self.total_samples,
            "success_samples": self.success_samples,
            "success_rate": self.success_rate
        }

    def __str__(self):
        return f"Model: {self.model_name:<21} | Attack Success: {self.success_rate:.1f}% ({self.success_samples}/{self.total_samples})"


def save_images(img_tensor, save_path):
    """
    Converts torch tensor back to image and saves it.
    """
    # Move to CPU, denormalize and scale [0, 1] -> [0, 255]
    img = np.array(img_tensor.cpu().detach().numpy()).transpose(1, 2, 0) * 255.
    img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[2] == 1:
        img = img.squeeze(axis=2)  # Remove the channel dimension for grayscale
    im = Image.fromarray(img)
    im.save(save_path)