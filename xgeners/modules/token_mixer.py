import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .utils import XGENERS_DEBUG, get_activation_fn, get_norm_fn


def get_token_mixer(token_mixer):
    if token_mixer == "softmax":
        return SoftmaxAttention
    elif token_mixer == "linear":
        return LinearAttention
    else:
        return SoftmaxAttention


class SoftmaxAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads=8,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        if XGENERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.num_heads = num_heads

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x):
        qkv = self.qkv_proj(x).chunk(3, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv
        )
        output = F.scaled_dot_product_attention(q, k, v)

        output = rearrange(output, "b h n d -> b n (h d)")

        output = self.output_proj(output)

        return output


class LinearAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads=8,
        norm_type="layernorm",
        act_fun="swish",
        use_lightnet=False,
        bias=False,
        use_lrpe=False,
        **kwargs,
    ):
        super().__init__()
        if XGENERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.num_heads = num_heads
        self.head_dim = embed_dim // self.num_heads

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.output_gate_proj = nn.Sequential(
            nn.Linear(embed_dim, self.head_dim, bias=bias),
            nn.Linear(self.head_dim, embed_dim, bias=bias),
        )

        self.norm = get_norm_fn(norm_type)(embed_dim)
        self.act_fun = get_activation_fn(act_fun)
        self.use_lightnet = use_lightnet

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv
        )
        q = self.act_fun(q)
        if self.use_lightnet:
            k = F.softmax(k, dim=-2)
        else:
            k = self.act_fun(k)

        # compute
        scale = q.shape[-1] ** 0.5
        kv = torch.matmul(k.transpose(-1, -2) / scale, v)
        output = torch.matmul(q / scale, kv)
        output = rearrange(output, "b h n d -> b n (h d)")
        output = self.norm(output)

        # output gate
        output_gate = F.sigmoid(self.output_gate_proj(x))
        output = output * output_gate

        # output projection
        output = self.output_proj(output)

        return output
