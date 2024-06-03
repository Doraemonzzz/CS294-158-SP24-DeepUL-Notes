import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VaeBaseline(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, image, **kwargs):
        b, c, h, w = image.shape
        image = rearrange(image, "b c h w -> b (c h w)")
        mu, logvar = self.encode(image.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        output = rearrange(output, "b (c h w) -> b c h w", h=h, w=w)

        output_dict = {"output": output, "mu": mu, "log_var": logvar}
        return output_dict

    def sample(self, num_samples):
        noise = torch.randn((num_samples, 20), device=torch.cuda.current_device())
        # samples = F.sigmoid(self.decode(noise, None))
        samples = self.decode(noise).view(-1, 1, 28, 28)

        return samples
