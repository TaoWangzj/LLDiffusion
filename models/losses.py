from torch import nn
from torch.nn.functional import l1_loss, mse_loss

from utils.transforms import inverse_data_transform


def noise_estimation_loss(model, x0, t, e, b):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    t = t.float()
    lr, hr = x0.chunk(2, dim=1)
    noisy_hr = hr * a.sqrt() + e * (1.0 - a).sqrt()
    pred_noise = model((noisy_hr, lr), t)
    noise_loss = mse_loss(pred_noise, e)
    return noise_loss


def twin_loss(model, x0, t, e, b):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    t = t.float()
    lr, hr = x0.chunk(2, dim=1)
    noisy_hr = hr * a.sqrt() + e * (1.0 - a).sqrt()
    pred_noise, pred_image, pred_color = model((noisy_hr, lr), t)

    gt = inverse_data_transform(hr)
    gt_color_map = (hr / (hr.sum(dim=1, keepdims=True) + 1e-4)).clip(0, 1)

    color_map_loss = l1_loss(pred_color, gt_color_map)
    image_loss = l1_loss(pred_image, gt)
    noise_loss = mse_loss(pred_noise, e)

    return {
        "noise_loss": noise_loss,
        "image_loss": image_loss,
        "color_loss": color_map_loss,
    }


def deg_loss(model, x0, t, e, b, epoch, config):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    t = t.float()
    lr, hr = x0.chunk(2, dim=1)
    noisy_hr = hr * a.sqrt() + e * (1.0 - a).sqrt()

    inputs = (noisy_hr, lr, hr)
    fixed_encoder = getattr(config.params, "fixed_encoder", False) # false
    degradation = getattr(config.params, "degradation", True) # false
    with_deg = degradation and not fixed_encoder
    if not with_deg:
        pred_noise = model(inputs, t)
        noise_loss = mse_loss(pred_noise, e)
        return noise_loss

    perceptual = "perceptual_loss" in vars(config.params.loss_weights)
    fix_epoch = getattr(config.params, "fix_epoch", None)
    fix_encoder = fix_epoch is not None and epoch >= fix_epoch

    if fix_encoder:
        module = model
        if isinstance(model, nn.DataParallel):
            module = model.module
        for p in module.encoder.parameters():
            if not p.requires_grad:
                break
            p.requires_grad = False

    if perceptual:
        pred_noise, pred_lr, p1, p2 = model(inputs, t, return_latents=True)
    else:
        pred_noise, pred_lr = model(inputs, t)

    noise_loss = mse_loss(pred_noise, e)

    if perceptual:
        preceptual_losses = [l1_loss(_1, _2) for _1, _2 in zip(p1, p2)]
        preceptual_loss = sum(preceptual_losses) / len(preceptual_losses)

    loss_dict = {
        "noise_loss": noise_loss,
    }

    if not fix_encoder:
        pred_lr = inverse_data_transform(pred_lr)
        lr = inverse_data_transform(lr)
        image_loss = l1_loss(pred_lr, lr)
        loss_dict["image_loss"] = image_loss
        if perceptual:
            loss_dict["perceptual_loss"] = preceptual_loss

    return loss_dict


def restore_former_loss(model, x0, t, noise, beta):
    cum_alpha = (1 - beta).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    t = t.float()
    lr, hr = x0.chunk(2, dim=1)
    noisy_hr = hr * cum_alpha.sqrt() + noise * (1.0 - cum_alpha).sqrt()
    output, diff = model((noisy_hr, lr), t)
    noise_loss = mse_loss(output, noise)
    codebook_loss = diff.mean()
    return {
        "noise_loss": noise_loss,
        "codebook_loss": codebook_loss,
    }


def compute_loss(self, x, t, e, b):
    model_name = getattr(self.config.model, "name", "UNet")
    if model_name == "TwinUNet":
        return twin_loss(self.model, x, t, e, b)
    elif model_name == "DegUNet":
        return deg_loss(self.model, x, t, e, b, self.epoch, self.config)
    elif model_name == "RestoreFormer":
        return restore_former_loss(self.model, x, t, e, b)
    return noise_estimation_loss(self.model, x, t, e, b)
