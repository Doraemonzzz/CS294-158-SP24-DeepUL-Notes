from .channel_mixer import get_channel_mixer
from .drop import DropPath
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
        self.token_norm = get_norm_fn(norm_type)(dim)
        self.feature_norm = get_norm_fn(norm_type)(dim)

    def forward(self, x):
        x = x + self.drop_path(self.token_mixer(self.token_norm(x)))
        x = x + self.drop_path(self.feature_mixer(self.feature_norm(x)))

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
            self.layers.append(
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
        return self.blocks(x)
