import os
import warnings

import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.io import read_image
from torchvision.transforms.functional import (
    convert_image_dtype,
    rgb_to_grayscale,
)


def correct_illumination(img1, img2):
    img1 = convert_image_dtype(img1, torch.float32)
    img2 = convert_image_dtype(img2, torch.float32)
    mean1 = img1.mean(dim=(1, 2), keepdim=True)
    mean2 = img2.mean(dim=(1, 2), keepdim=True)
    scale = mean2 / mean1
    img1 = img1 * scale
    img1 = img1.clip(0.0, 1.0)
    return img1


def as_numpy(img):
    img = convert_image_dtype(img, torch.uint8)
    img = img.permute(1, 2, 0).cpu().numpy()
    return img


def calculate_psnr(img1, img2):
    img1 = as_numpy(img1)
    img2 = as_numpy(img2)
    psnr = peak_signal_noise_ratio(img1, img2)
    return psnr


def calculate_ssim(img1, img2, y_channels=False):
    if y_channels:
        img1 = rgb_to_grayscale(img1, num_output_channels=3)
        img2 = rgb_to_grayscale(img2, num_output_channels=3)

    img1 = as_numpy(img1)
    img2 = as_numpy(img2)
    ssim = structural_similarity(img1, img2, channel_axis=2)
    return ssim


def calculate_lpips(img1, img2):
    img1 = convert_image_dtype(img1, torch.float32)
    img2 = convert_image_dtype(img2, torch.float32)
    imgs1 = img1.unsqueeze(0)
    imgs2 = img2.unsqueeze(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metric = LearnedPerceptualImagePatchSimilarity(normalize=True)
        lpips = metric(imgs1, imgs2)
    return lpips.item()


def metric(img1, img2, correct=True, ssim_y_channels=False):
    if correct:
        img1 = correct_illumination(img1, img2)

    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2, y_channels=ssim_y_channels)
    lpips = calculate_lpips(img1, img2)
    return {
        "PSNR": psnr,
        "SSIM": ssim,
        "LPIPS": lpips,
    }


def evaluate(x_path, y_path, correct=True, ssim_y_channels=False, verbose=False):
    x_files = sorted(os.listdir(x_path))
    y_files = sorted(os.listdir(y_path))
    assert len(x_files) == len(y_files)

    x_files = [os.path.join(x_path, f) for f in x_files]
    y_files = [os.path.join(y_path, f) for f in y_files]

    scores_dict = {}
    for i in range(len(x_files)):
        res = read_image(x_files[i])
        gt = read_image(y_files[i])
        score_dict = metric(res, gt, correct, ssim_y_channels)
        for k, v in score_dict.items():
            scores_dict.setdefault(k, []).append(v)
        if verbose:
            name = os.path.basename(x_files[i])
            print(f"{name}: " + ", ".join(
                "{k}:{v:.4}".format(k=k, v=v)
                for k, v in score_dict.items()
            ))

    scores_dict = {k: sum(v) / len(v) for k, v in scores_dict.items()}
    return scores_dict
