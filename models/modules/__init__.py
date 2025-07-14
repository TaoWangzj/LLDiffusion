from copy import copy

import torch

from .dit import DiT
from .unet import DiffusionUNet


def get_model(config):
    model_name = getattr(config.model, "name", "UNet")
    if model_name == "UNet":
        model = DiffusionUNet(config)
    elif model_name == "DiT":
        model_config = vars(config.model)
        model_config = copy(model_config)
        model_config.pop("name")
        model = DiT(**model_config)
    elif model_name == "TwinUNet":
        from .twin_unet import TwinUNet
        model = TwinUNet(config)
    elif model_name == "DegUNet":
        from .degradation import DegrationUNet
        model = DegrationUNet(config)
    elif model_name == "RestoreFormer":
        from .restore_former import RestoreFormer
        model_config = vars(config.model)
        model_config = copy(model_config)
        model_config.pop("name")
        model = RestoreFormer(**model_config)
    elif model_name == "Restormer":
        from .resformer import Restormer
        model_config = vars(config.model)
        model_config = copy(model_config)
        model_config.pop("name")
        model = Restormer(**model_config)
    elif model_name == "SR3":
        from .sr3 import UNet
        model = UNet(in_channel=config.model.in_channels)
    else:
        raise NotImplementedError(model_name)

    # preprocess conditional inputs
    est = getattr(config.data, "est", False)
    if est:
        from models.estimator import Estimator
        model.estimator = Estimator()

    maps = getattr(config.data, "maps", [])

    if model_name in [
        "UNet", "DiT", "TwinUNet", "RestoreFormer", "Restormer", "SR3"
    ] or est:
        def forward_pre_hook(module, inputs):
            (noisy_x, cond_x), t = inputs
            if cond_x is not None:
                if len(maps) != 0:
                    from utils.maps import compute_conditional_inputs
                    noisy_x = compute_conditional_inputs(cond_x, noisy_x, indices=maps)
                elif est:
                    unnorm_cond = (cond_x + 1) / 2
                    color_map = module.estimator(unnorm_cond)
                    norm_color_map = color_map * 2 - 1
                    noisy_x = torch.cat([noisy_x, cond_x, norm_color_map], dim=1)
                else:
                    noisy_x = torch.cat([noisy_x, cond_x], dim=1)
                    return noisy_x, t
            return noisy_x, t
        model.register_forward_pre_hook(forward_pre_hook)

    model = torch.nn.DataParallel(model).to(config.device)
    return model
