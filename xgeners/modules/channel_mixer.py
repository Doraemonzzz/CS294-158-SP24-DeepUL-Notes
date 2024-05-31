import torch.nn as nn

from .utils import XGENERS_DEBUG, get_activation_fn, print_params


def get_channel_mixer(channel_mixer):
    if channel_mixer == "glu":
        return GLU
    elif channel_mixer == "ffn":
        return FFN
    else:
        return GLU


class FFN(nn.Module):
    def __init__(self, embed_dim: int, mid_dim: int, act_fun: str, bias: bool = False):
        super().__init__()
        if XGENERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.w1 = nn.Linear(embed_dim, mid_dim, bias=bias)
        self.w2 = nn.Linear(mid_dim, embed_dim, bias=bias)
        self.act_fun = get_activation_fn(act_fun)

    def forward(self, x):
        output = self.w2(self.act_fun(self.w1(x)))

        return output


class GLU(nn.Module):
    def __init__(
        self, embed_dim: int, mid_dim: int, act_fun: str, bias: bool = False
    ) -> None:
        super().__init__()

        if XGENERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.w1 = nn.Linear(embed_dim, mid_dim, bias=bias)
        self.w2 = nn.Linear(embed_dim, mid_dim, bias=bias)
        self.w3 = nn.Linear(mid_dim, embed_dim, bias=bias)
        self.act = get_activation_fn(act_fun)

    def forward(self, x):
        output = self.w3(self.act(self.w1(x)) * self.w2(x))

        return output
