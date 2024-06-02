import abc

from torch import nn


class GenModel(abc.ABC, nn.Module):
    @abc.abstractmethod
    def sample(self, num_samples):
        ...


class AE(GenModel):
    @abc.abstractmethod
    def encode(self, x):
        ...

    @abc.abstractmethod
    def decode(self, x):
        ...
