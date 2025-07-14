import torch

from torchvision.transforms.functional import equalize, convert_image_dtype


def compute_maps(x):
    x = x.clip(0, 1)
    hist_img = equalize(convert_image_dtype(x, torch.uint8))
    hist_img = hist_img / 255

    color_map = x / (x.sum(dim=1, keepdims=True) + 1e-4)
    color_map = color_map.clip(0, 1)

    def gradient(x):

        def sub_gradient(x):
            left_shift_x, right_shift_x, grad = torch.zeros_like(
                x), torch.zeros_like(x), torch.zeros_like(x)
            left_shift_x[:, :, 0:-1] = x[:, :, 1:]
            right_shift_x[:, :, 1:] = x[:, :, 0:-1]
            grad = 0.5 * (left_shift_x - right_shift_x)
            return grad

        return sub_gradient(x), sub_gradient(torch.transpose(x, 2, 3)).transpose(2, 3)

    dx, dy = gradient(color_map)
    noise_map = torch.max(torch.stack([dx.abs(), dy.abs()], dim=0), dim=0)[0]
    return hist_img, color_map, noise_map


def normalize(x):
    return 2 * x - 1


def unnormalize(x):
    return (x + 1) / 2


def compute_conditional_inputs(lr, hr, norm=True, indices=(1,)):
    if len(indices) == 0:
        return torch.cat([lr, hr], dim=1)

    if norm:
        lr = unnormalize(lr)
        hr = unnormalize(hr)

    feature_maps = compute_maps(lr)
    feature_maps = list(feature_maps)
    if norm:
        lr = normalize(lr)
        hr = normalize(hr)
        feature_maps[0] = normalize(feature_maps[0])
        feature_maps[1] = normalize(feature_maps[1])
        feature_maps[2] = normalize(feature_maps[2])
    feature_maps = [feature_maps[i] for i in indices]

    return torch.cat([lr, hr, *feature_maps], dim=1)
