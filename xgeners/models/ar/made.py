# https://arxiv.org/pdf/1502.03509.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones((out_features, in_features)))

    def set_mask(self, mask):
        self.mask.data.copy_(mask)

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dims,
        random_order=False,
        bias=True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = in_dim
        self.random_order = random_order

        dims = [self.in_dim] + hidden_dims + [self.out_dim]
        n = len(dims)
        self.layers = nn.ModuleList([])
        for i in range(n - 2):
            d_in = dims[i]
            d_out = dims[i + 1]
            self.layers.append(MaskedLinear(d_in, d_out, bias))
            self.layers.append(nn.ReLU())

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
            index_list.append(torch.randint(min, max, (dims[i],)))

        ##### construct mask
        for i in range(n - 2):
            mask = index_list[i][None, :] <= index_list[i + 1][:, None]
            self.layers[2 * i].set_mask(mask)

        mask = index_list[n - 2][None, :] < index_list[n - 1][:, None]
        # print(len(self.layers), 2 * n - 2)
        self.layers[2 * n - 4].set_mask(mask)

    def forward(self, x):
        x = rearrange(x, "b d h w -> b d (h w)")
        for layer in self.layers:
            x = layer(x)
        return x


def reproduce(
    n_epochs=85,
    batch_size=64,
    log_dir="/tmp/run",
    n_gpus=1,
    device_id=0,
    debug_loader=None,
):
    """Training script with defaults to reproduce results.

    The code inside this function is self contained and can be used as a top level
    training script, e.g. by copy/pasting it into a Jupyter notebook.

    Args:
        n_epochs: Number of epochs to train for.
        batch_size: Batch size to use for training and evaluation.
        log_dir: Directory where to log trainer state and TensorBoard summaries.
        n_gpus: Number of GPUs to use for training the model. If 0, uses CPU.
        device_id: The device_id of the current GPU when training on multiple GPUs.
        debug_loader: Debug DataLoader which replaces the default training and
            evaluation loaders if not 'None'. Do not use unless you're writing unit
            tests.
    """
    from torch import optim
    from torch.nn import functional as F

    from xgeners import datasets, models, trainer

    train_loader, test_loader = debug_loader, debug_loader
    if train_loader is None:
        train_loader, test_loader = datasets.get_mnist_loaders(
            batch_size, dynamically_binarize=True
        )

    model = models.MADE(in_dim=784, hidden_dims=[8000])
    print(model)
    optimizer = optim.Adam(model.parameters())

    def loss_fn(x, _, preds):
        batch_size = x.shape[0]
        x, preds = x.view((batch_size, -1)), preds.view((batch_size, -1))
        loss = F.binary_cross_entropy_with_logits(preds, x, reduction="none")
        return loss.sum(dim=1).mean()

    model_trainer = trainer.Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=test_loader,
        log_dir=log_dir,
        n_gpus=n_gpus,
        device_id=device_id,
    )
    model_trainer.interleaved_train_and_eval(n_epochs)
