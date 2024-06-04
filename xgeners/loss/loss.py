import torch
from torch.nn import functional as F


def vae_loss(image, output, mu, log_var, **kwargs):
    recons_loss = F.mse_loss(image, output)
    kl_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=-1), dim=0
    )
    kl_weight = kwargs["kl_weight"]

    loss = recons_loss + kl_weight * kl_loss
    return loss


def binary_vae_loss(image, output, mu, log_var, **kwargs):
    b = image.shape[0]
    recons_loss = F.binary_cross_entropy_with_logits(output, image, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp())
    kl_weight = kwargs["kl_weight"]
    loss = (recons_loss + kl_weight * kl_loss) / b
    return loss


def ae_loss(image, output, **kwargs):
    loss = F.mse_loss(image, output)
    return loss


def binary_ae_loss(image, output, **kwargs):
    loss = F.binary_cross_entropy_with_logits(output, image)
    return loss
