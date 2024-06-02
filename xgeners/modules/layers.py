import torch
import torch.nn as nn

from .channel_mixer import get_channel_mixer
from .drop import DropPath
from .patch_embed import PatchEmbed, ReversePatchEmbed
from .pe import get_pe
from .token_mixer import get_token_mixer
from .utils import get_norm_fn


class Block(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        mid_dim,
        token_mixer="softmax",
        channel_mixer="glu",
        drop_path=0.0,
        norm_type="layernorm",
        act_fun="swish",
        use_lightnet=False,
        bias=False,
        use_lrpe=False,
        **kwargs
    ):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.token_mixer = get_token_mixer(token_mixer)(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            norm_type=norm_type,
            act_fun=act_fun,
            use_lightnet=use_lightnet,
            use_lrpe=use_lrpe,
        )
        self.channel_mixer = get_channel_mixer(channel_mixer)(
            embed_dim=embed_dim,
            mid_dim=mid_dim,
            act_fun=act_fun,
            bias=bias,
        )
        self.token_norm = get_norm_fn(norm_type)(embed_dim)
        self.channel_norm = get_norm_fn(norm_type)(embed_dim)

    def forward(self, x):
        x = x + self.drop_path(self.token_mixer(self.token_norm(x)))
        x = x + self.drop_path(self.channel_mixer(self.channel_norm(x)))

        return x


class Blocks(nn.Module):
    def __init__(
        self,
        num_layers,
        embed_dim,
        num_heads,
        mid_dim,
        token_mixer="softmax",
        channel_mixer="glu",
        drop_path=0.0,
        norm_type="layernorm",
        act_fun="swish",
        use_lightnet=False,
        bias=False,
        use_lrpe=False,
        **kwargs
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_layers):
            self.blocks.append(
                Block(
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
                    **kwargs
                )
            )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        channels,
        flatten,
        num_layers,
        embed_dim,
        num_heads,
        mid_dim,
        token_mixer="softmax",
        channel_mixer="glu",
        drop_path=0.0,
        norm_type="layernorm",
        act_fun="swish",
        use_lightnet=False,
        bias=False,
        use_lrpe=False,
        pe_name="sincos_2d",
        **kwargs
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            channels=channels,
            flatten=flatten,
            bias=bias,
        )
        self.pe = get_pe(
            pe_name,
            embed_dim=embed_dim,
            grid_size=(self.patch_embed.num_h_patch, self.patch_embed.num_w_patch),
        )
        self.blocks = Blocks(
            num_layers=num_layers,
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

    def forward(self, x, extra_tokens=None):
        x = self.pe(self.patch_embed(x))
        offset = 0
        if extra_tokens != None:
            x = torch.cat([extra_tokens, x], dim=-2)
            offset = extra_tokens.shape[-2]
        x = self.blocks(x)[:, offset:]

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        channels,
        flatten,
        num_layers,
        embed_dim,
        num_heads,
        mid_dim,
        token_mixer="softmax",
        channel_mixer="glu",
        drop_path=0.0,
        norm_type="layernorm",
        act_fun="swish",
        use_lightnet=False,
        bias=False,
        use_lrpe=False,
        pe_name="sincos_2d",
        **kwargs
    ):
        super().__init__()
        self.reverse_patch_embed = ReversePatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            channels=channels,
            flatten=flatten,
            bias=bias,
        )
        self.pe = get_pe(
            pe_name,
            embed_dim=embed_dim,
            grid_size=(
                self.reverse_patch_embed.num_h_patch,
                self.reverse_patch_embed.num_w_patch,
            ),
        )
        self.blocks = Blocks(
            num_layers=num_layers,
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

    def forward(self, x, extra_tokens=None):
        x = self.pe(x)
        offset = 0
        if extra_tokens != None:
            x = torch.cat([extra_tokens, x], dim=-2)
            offset = extra_tokens.shape[-2]
        x = self.blocks(x)[:, offset:]
        x = self.reverse_patch_embed(x)

        return x
