from einops import rearrange
from torch import nn

from .utils import pair


class PatchEmbed(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        embed_dim,
        channels=3,
        flatten=True,
        bias=False,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        channels * patch_height * patch_width

        self.to_patch_embedding = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )

        self.flatten = flatten
        self.num_h_patch = image_height // patch_height
        self.num_w_patch = image_width // patch_width
        self.num_tokens = self.num_h_patch * self.num_w_patch

    def forward(self, x):
        y = self.to_patch_embedding(x)
        if self.flatten:
            y = rearrange(y, "b c h w -> b (h w) c")

        return y


class ReversePatchEmbed(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        embed_dim,
        channels=3,
        flatten=True,
        bias=False,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        channels * patch_height * patch_width

        self.reverse_patch_embedding = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.flatten = flatten
        self.num_h_patch = image_height // patch_height
        self.num_w_patch = image_width // patch_width
        self.num_tokens = self.num_h_patch * self.num_w_patch

    def forward(self, x):
        if self.flatten:
            x = rearrange(x, "b (h w) c -> b c h w", h=self.num_h_patch)

        y = self.reverse_patch_embedding(x)
        return y
