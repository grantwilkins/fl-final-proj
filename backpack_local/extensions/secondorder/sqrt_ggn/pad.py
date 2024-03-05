"""Contains ``SqrtGGN{Exact, MC}`` extension for BackPACK's custom ``Pad`` module."""

from backpack_local.core.derivatives.pad import PadDerivatives
from backpack_local.extensions.secondorder.sqrt_ggn.base import SqrtGGNBaseModule


class SqrtGGNPad(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``backpack.custom_modules.pad.Pad``."""

    def __init__(self):
        """Pass derivatives for ``backpack.custom_modules.pad.Pad`` module."""
        super().__init__(PadDerivatives())
