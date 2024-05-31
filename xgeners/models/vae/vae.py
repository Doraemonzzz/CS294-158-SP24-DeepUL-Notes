import torch
from torch import nn
from torch.nn import functional as F

from xgeners.modules import Blocks, PatchEmbed, ReversePatchEmbed


class VanillaVAE(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        kwargs["in_channels"]
        latent_dim = kwargs["latent_dim"]
        kwargs.get("hidden_dims", None)

        # Encoder part
        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            channels=channels,
            flatten=flatten,
            bias=bias,
        )
        self.encoder = Blocks(
            num_layers=num_encoder_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mid_dim=mid_dim,
            token_mixer=token_mixer,
            channel_mixer=channel_mixer,
            drop_path=drop_path,
            norm_type=norm_type,
            act_fun=act_fun,
            use_lightnet=use_lightnet,
            bias=bias,
            use_lrpe=use_lrpe,
        )
        self.mu_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.log_var_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = None
        self.mu_proj = nn.Linear(embed_dim, latent_dim, bias=bias)
        self.log_var_proj = nn.Linear(embed_dim, latent_dim, bias=bias)

        # Decoder part
        self.decoder_proj = nn.Linear(latent_dim, embed_dim, bias=bias)
        self.decoder = Blocks(
            num_layers=num_decoder_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mid_dim=mid_dim,
            token_mixer=token_mixer,
            channel_mixer=channel_mixer,
            drop_path=drop_path,
            norm_type=norm_type,
            act_fun=act_fun,
            use_lightnet=use_lightnet,
            bias=bias,
            use_lrpe=use_lrpe,
        )
        self.reverse_patch_embed = ReversePatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            channels=channels,
            flatten=flatten,
            bias=bias,
        )
        # ref https://github.com/chinmay5/vit_ae_plus_plus/blob/main/model/vit_autoenc.py
        self.dummy_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def encode(self, x: Tensor) -> List[Tensor]:
        x = self.to_patch_embedding(x)
        b = x.shape[0]
        mu_token = self.mu_token.expand(b, -1, -1)
        log_var_token = self.log_var_token.expand(b, -1, -1)
        x = torch.cat([mu_token, log_var_token, x], dim=-2)
        x = self.encoder(x)
        mu_token = x[:, 0]
        log_var_token = x[:, 1]
        mu = self.mu_proj(mu_token)
        log_var = self.log_var_token(log_var_token)

        return [mu, log_var]

    def decode(self, x, n) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        x = self.decoder_proj(x)
        dummy_tokens = self.mask_token.repeat(x.shape[0], n, 1)
        x = torch.cat([x, dummy_tokens], dim=-2)
        y = self.decoder(x)
        y = self.reverse_patch_embed(y)
        return y[:, 1:]

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        output = self.decode(z)

        output_dict = {"output": output, "mu": mu, "log_var": log_var}
        return output_dict

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": -kld_loss.detach(),
        }

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)["output"]
