"""Define our models, and training and eval functions."""

from torch import nn

from project.types.common import IsolatedRNG
from torchvision.models import resnet18


class Net(nn.Module):
    """A PyTorch model."""

    def __init__(self, pretrained: bool = False, **kwargs):
        super(Net, self).__init__()
        self.model = resnet18(pretrained=pretrained, **kwargs)

    def forward(self, x):
        return self.model(x)


def get_net(_config: dict, rng_tuple: IsolatedRNG) -> nn.Module:
    """Return a model instance.

    Args:
    config: A dictionary with the model configuration.
    rng_tuple: The random number generator state for the training.
        Use if you need seeded random behavior

    Returns
    -------
    nn.Module
        A PyTorch model.
    """
    return Net(pretrained=False)
