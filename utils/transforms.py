import torch

from torchvision.transforms.functional import convert_image_dtype

def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def torch_to_cv(x):
    x = convert_image_dtype(x, torch.uint8)
    x = x[[2, 1, 0], :, :].permute(1, 2, 0)
    x = x.cpu().numpy()
    return x
