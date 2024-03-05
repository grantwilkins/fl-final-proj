"""Contains extensions for the flatten layer used by ``SqrtGGN{Exact, MC}``."""

from backpack_local.core.derivatives.flatten import FlattenDerivatives
from backpack_local.extensions.secondorder.sqrt_ggn.base import SqrtGGNBaseModule


class SqrtGGNFlatten(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.Flatten`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.Flatten`` module."""
        super().__init__(FlattenDerivatives())
