from einops import rearrange
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        channels=3,
        flatten=True,
        bias=False,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Conv2d(
            channels, patch_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )

        self.flatten = flatten

    def forward(self, x):
        y = self.to_patch_embedding(x)
        if self.flatten:
            y = rearrange(y, "b h w d -> b (h w) d")

        return y


class ReversePatchEmbed(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        channels=3,
        flatten=True,
        bias=False,
    ):
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        patch_dim = channels * patch_height * patch_width

        self.reverse_patch_embedding = nn.ConvTranspose2d(
            in_channels=patch_dim,
            out_channels=channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        return self.reverse_patch_embedding(x)
