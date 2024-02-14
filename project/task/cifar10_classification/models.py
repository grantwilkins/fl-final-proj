"""Define our models, and training and eval functions."""

from torch import nn, Tensor

from torchvision.models import resnet18
from project.utils.utils import lazy_config_wrapper
from project.types.common import NetGen


class Net(nn.Module):
    """A PyTorch model."""

    def __init__(self, pretrained: bool = False) -> None:
        """Initialize the model.

        Args:
        pretrained: Whether to use a pretrained model.
        """
        super().__init__()
        self.model = resnet18(pretrained=pretrained)

    def forward(self, x: Tensor) -> Tensor:
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
