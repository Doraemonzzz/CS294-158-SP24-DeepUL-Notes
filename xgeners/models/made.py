# https://arxiv.org/pdf/1502.03509.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.mask = torch.empty(0)

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dims,
        out_dim,
        random_order=False,
        bias=True,
    ):
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.random_order = random_order

        dims = [in_dim] + hidden_dims + [out_dim]
        n = len(dims)
        self.layers = nn.ModuleList([])
        for i in range(n - 2):
            d_in = dims[i]
            d_out = dims[i + 1]
            self.layers.append(MaskedLinear(d_in, d_out, bias), nn.ReLU())

        self.layers.append(
            MaskedLinear(dims[n - 2], dims[n - 1], bias),
        )
        self.random_order = random_order
        self.set_mask()

    def set_mask(self):
        ##### construct index
        index_list = []
        index = torch.arange(self.in_dim)
        if self.random_order:
            index = torch.permute(index)

        index_list.append(index)

        dims = [self.in_dim] + self.hidden_dims + [self.out_dim]
        n = len(dims)
        max = self.in_dim - 1
        for i in range(1, n):
            min = index_list[i - 1].min().item()
            index_list.append(torch.randint(min, max, dims[i]))

        ##### construct mask
        for i in range(n - 1):
            mask = index_list[i][:, None] <= index_list[i + 1][None, :]
            self.layers[2 * i].set_mask(mask)

        mask = index_list[n - 2][:, None] < index_list[n - 1][None, :]
        self.layers[2 * n - 2].set_mask(mask)

    def forward(self, x):
        return self.layers(x)
