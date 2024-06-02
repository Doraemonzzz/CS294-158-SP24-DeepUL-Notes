from torch import nn

from xgeners.models import AE
from xgeners.modules import ClassEmbed, Decoder, Encoder


class AutoEncoder(AE):
    def __init__(self, config) -> None:
        super().__init__()
        ##### get params start
        # patch embedding
        image_size = config.image_size
        patch_size = config.patch_size
        channels = config.channels
        flatten = config.flatten
        # class embedding
        num_class = config.num_class
        self.use_class_label = config.use_class_label

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

        # class embedding
        self.class_embed = ClassEmbed(num_class, embed_dim)

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

    def encode(self, x, extra_tokens):
        x_encode = self.encoder_proj(self.encoder(x, extra_tokens))
        return x_encode

    def decode(self, x_encode, extra_tokens):
        output = self.decoder(self.decoder_proj(x_encode), extra_tokens)

        return output

    def forward(self, image, **kwargs):
        label = kwargs.get("label", None)
        if self.use_class_label:
            extra_tokens = self.class_embed(label)
        else:
            extra_tokens = self.class_embed(None)
        extra_tokens = extra_tokens.repeat(image.shape[0], 1, 1)
        x_encode = self.encode(image, extra_tokens)
        output = self.decode(x_encode, extra_tokens)
        output_dict = {"output": output}
        return output_dict

    def sample(self, num_samples):
        return None
