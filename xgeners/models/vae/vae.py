import torch
import torch.nn.functional as F
from torch import nn

from xgeners.modules import AE, Decoder, Encoder


class VanillaVAE(AE):
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
        self.encoder_proj = nn.Linear(embed_dim, 2 * latent_dim, bias=bias)

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

        # others
        self.num_patch = self.encoder.patch_embed.num_patch
        self.latent_dim = latent_dim

    def encode(self, x):
        x = self.encoder(x)
        mu, log_var = self.encoder_proj(x).chunk(2, dim=-1)
        return [mu, log_var]

    def decode(self, x):
        x = self.decoder_proj(x)
        y = self.decoder(x)

        return y

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, image, **kwargs):
        mu, log_var = self.encode(image)
        z = self.reparameterize(mu, log_var)
        output = self.decode(z)
        output_dict = {"output": output, "mu": mu, "log_var": log_var}
        return output_dict

    def sample(self, num_samples):
        noise = torch.randn(
            (num_samples, self.num_patch, self.latent_dim),
            device=torch.cuda.current_device(),
        )

        samples = F.sigmoid(self.decode(noise))
        return samples
