import torch
import torch.nn as nn


class ClassEmbed(nn.Module):
    def __init__(
        self,
        num_class,
        embed_dim,
    ):
        super().__init__()
        self.num_class = num_class
        self.embed = nn.Embedding(num_class + 1, embed_dim)

    def forward(self, label):
        if label == None:
            label = torch.tensor([self.num_class], device=torch.cuda.current_device())
            return self.embed(label)
        else:
            return self.embed(label)
