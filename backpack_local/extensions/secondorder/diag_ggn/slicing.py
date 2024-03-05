"""Holds ``DiagGGN{Exact, MC}`` extension for BackPACK's custom ``Slicing`` module."""

from backpack_local.core.derivatives.slicing import SlicingDerivatives
from backpack_local.extensions.secondorder.diag_ggn.diag_ggn_base import (
    DiagGGNBaseModule,
)


class DiagGGNSlicing(DiagGGNBaseModule):
    """``DiagGGN{Exact, MC}`` for ``backpack.custom_modules.slicing.Slicing``."""

    def __init__(self):
        """Pass derivatives for ``backpack.custom_modules.pad.Pad`` module."""
        super().__init__(SlicingDerivatives())
