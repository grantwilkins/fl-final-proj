"""Define our models, and training and eval functions."""

import torch
from torch import nn

from torchvision.models import resnet18
from project.utils.utils import lazy_config_wrapper
from project.types.common import NetGen


class Net(nn.Module):
    """A PyTorch model."""

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize the model.

        Args:
        num_classes: How many classes in the subsequent model.
        """
        super().__init__()
        self.model = resnet18(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
        x: The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        return self.model(x)


get_net: NetGen = lazy_config_wrapper(Net)
