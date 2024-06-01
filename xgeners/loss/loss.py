import torch
from torch.nn import functional as F


def vae_loss(input, output, mu, log_var, **kwargs):
    recons_loss = F.mse_loss(output, input)
    kl_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=-1), dim=0
    )
    kl_weight = kwargs["kl_weight"]

    loss = recons_loss + kl_weight * kl_loss
    return loss
