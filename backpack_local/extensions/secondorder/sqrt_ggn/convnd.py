"""Contains extensions for convolution layers used by ``SqrtGGN{Exact, MC}``."""

from backpack_local.core.derivatives.conv1d import Conv1DDerivatives
from backpack_local.core.derivatives.conv2d import Conv2DDerivatives
from backpack_local.core.derivatives.conv3d import Conv3DDerivatives
from backpack_local.extensions.secondorder.sqrt_ggn.base import SqrtGGNBaseModule


class SqrtGGNConv1d(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.Conv1d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.Conv1d`` module."""
        super().__init__(Conv1DDerivatives(), params=["bias", "weight"])


class SqrtGGNConv2d(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.Conv2d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.Conv2d`` module."""
        super().__init__(Conv2DDerivatives(), params=["bias", "weight"])


class SqrtGGNConv3d(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.Conv3d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.Conv3d`` module."""
        super().__init__(Conv3DDerivatives(), params=["bias", "weight"])
