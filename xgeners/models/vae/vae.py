import torch
from torch import nn


class VanillaVAE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        ##### get params start
        # patch embedding
        image_size = config.image_size
        patch_size = config.patch_size
        channels = config.channels
        flatten = config.flatten
        # model params
        latent_dim = config.latent_dim
        embed_dim = config.embed_dim
        num_heads = config.num_heads
        bias = config.bias
        mid_dim = config.mid_dim  # glu/ffn mid dim
        token_mixer = config.token_mixer  # "softmax" or "linear"
        channel_mixer = config.channel_mixer  # "glu" or "ffn"
        drop_path = config.drop_path  # drop path rate
        norm_type = config.norm_type
        act_fun = config.act_fun
        use_lightnet = config.use_lightnet
        use_lrpe = config.use_lrpe
        pe_name = config.pe_name

        # encoder
        num_encoder_layers = config.num_encoder_layers

        # decoder
        num_decoder_layers = config.num_decoder_layers
        ##### get params end

        # Encoder part
        self.encoder = Encoder(
            image_size=image_size,
            patch_size=patch_size,
            channels=channels,
            flatten=flatten,
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
            pe_name=pe_name,
        )
        self.encoder_proj = nn.Linear(embed_dim, latent_dim, bias=bias)

        self.mu_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.log_var_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = None
        self.mu_proj = nn.Linear(embed_dim, latent_dim, bias=bias)
        self.log_var_proj = nn.Linear(embed_dim, latent_dim, bias=bias)

        # Decoder part
        self.decoder_proj = nn.Linear(latent_dim, embed_dim, bias=bias)
        self.decoder = Decoder(
            image_size=image_size,
            patch_size=patch_size,
            channels=channels,
            flatten=flatten,
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
            pe_name=pe_name,
        )

        self.decoder_token = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_tokens, embed_dim)
        )

    def encode(self, x):
        x = self.patch_embed(x)
        b = x.shape[0]
        mu_token = self.mu_token.expand(b, -1, -1)
        log_var_token = self.log_var_token.expand(b, -1, -1)
        x = torch.cat([mu_token, log_var_token, x], dim=-2)
        x = self.encoder(x)
        mu_token = x[:, :1]
        log_var_token = x[:, 1:2]
        mu = self.mu_proj(mu_token)
        log_var = self.log_var_proj(log_var_token)
        return [mu, log_var]

    def decode(self, x):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        x = self.decoder_proj(x)
        decoder_token = self.decoder_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([x, decoder_token], dim=-2)
        y = self.decoder(x)
        y = self.reverse_patch_embed(y[:, 1:])
        return y

    def reparameterize(self, mu, logvar):
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

    def forward(self, image, **kwargs):
        mu, log_var = self.encode(image)
        z = self.reparameterize(mu, log_var)
        output = self.decode(z)
        output_dict = {"output": output, "mu": mu, "log_var": log_var}
        return output_dict

    def sample(self, num_samples: int, current_device: int, **kwargs):
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

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)["output"]
